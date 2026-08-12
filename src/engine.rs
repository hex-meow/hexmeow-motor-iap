use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use can_transport::{CanBus, CanFilter, CanFrame, CanId, CanRx};
use thiserror::Error;

use crate::{
    Frame, FrameAssembler, FunctionCode, IapIdentity, IapPolicy, ImgArtifact, PolicyError,
    ReadyToFlash, RegisteredTarget, SupportPolicy, MAX_DATA_LEN,
};

pub(crate) const TX_BASE: u16 = 0x780;
pub(crate) const RX_BASE: u16 = 0x680;

#[derive(Debug, Clone)]
pub struct CancellationToken {
    cancelled: Arc<AtomicBool>,
}

impl CancellationToken {
    pub fn new() -> Self {
        Self {
            cancelled: Arc::new(AtomicBool::new(false)),
        }
    }

    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::SeqCst);
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::SeqCst)
    }
}

impl Default for CancellationToken {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlashStage {
    Resetting,
    EnteringBootloader,
    IdentityVerified,
    StartingDownload,
    Transferring,
    Finalizing,
    Verifying,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlashEvent {
    Stage(FlashStage),
    IapIdentity(IapIdentity),
    Progress { written: usize, total: usize },
}

#[derive(Debug, Clone)]
pub struct FlashOptions {
    /// Quiet time after `Reset_Ack` before the first Enter-IAP knock. Purely an
    /// optimization: knocking during the reboot is harmless but wasted.
    pub knock_start_delay: Duration,
    /// Gap between Enter-IAP knocks, which is also how long each knock listens
    /// for its answer.
    pub knock_interval: Duration,
    /// How long to keep knocking, measured from the first knock. Must outlast
    /// the bootloader's own timeout so a missed window is reported as such
    /// rather than mistaken for a dead device.
    pub knock_window: Duration,
    pub reset_timeout: Duration,
    pub start_timeout: Duration,
    pub segment_timeout: Duration,
    pub final_timeout: Duration,
    pub verify_timeout: Duration,
    pub reset_attempts: u8,
    /// CAN frames to hand the transport before pausing for `tx_burst_pause`.
    pub tx_burst_frames: usize,
    pub tx_burst_pause: Duration,
}

impl Default for FlashOptions {
    fn default() -> Self {
        Self {
            // Measured on Meow Motor (0x4859444C/0xAAAA0001, revisions 9 and
            // 10): the bootloader starts answering Enter_Iap_Req ~1184 ms after
            // Reset_Req and jumps to the application ~2397 ms after it, so the
            // window is only ~1.2 s wide and does not contain the 700 ms single
            // shot the vendor documents. Knocking from 400 ms to 3500 ms covers
            // it with wide margin at both ends.
            knock_start_delay: Duration::from_millis(400),
            knock_interval: Duration::from_millis(25),
            knock_window: Duration::from_millis(3500),
            reset_timeout: Duration::from_secs(3),
            start_timeout: Duration::from_secs(20),
            segment_timeout: Duration::from_secs(3),
            final_timeout: Duration::from_secs(20),
            verify_timeout: Duration::from_secs(20),
            reset_attempts: 2,
            // 15 frames per millisecond asks for 15000 frames/s. An 8-byte
            // standard frame costs 111-130 bits with stuffing, so a 1 Mbit/s
            // link carries at most ~7700-9000 frames/s even when nothing else
            // is talking: the old default overdrove the bus roughly twofold and
            // leaned on the queue to absorb it, and a CAN queue is shallow
            // (txqueuelen defaults to 10, gs_usb adds ~10 in-flight URBs). Four
            // frames per millisecond stays well under line rate and spreads the
            // load instead of bursting into a queue that is already full. An
            // update takes seconds either way.
            tx_burst_frames: 4,
            tx_burst_pause: Duration::from_millis(1),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FlashOutcome {
    iap_identity: IapIdentity,
}

impl FlashOutcome {
    pub fn iap_identity(self) -> IapIdentity {
        self.iap_identity
    }
}

#[derive(Debug, Error)]
pub enum FlashError {
    #[error(transparent)]
    Policy(#[from] PolicyError),
    #[error("cancelled during {stage:?}; recovery_required={recovery_required}")]
    Cancelled {
        stage: FlashStage,
        recovery_required: bool,
    },
    #[error(
        "CAN transport failed during {stage:?}: {detail}; recovery_required={recovery_required}"
    )]
    Transport {
        stage: FlashStage,
        detail: String,
        recovery_required: bool,
    },
    #[error("timed out waiting for an unambiguous ACK during {stage:?}; recovery_required={recovery_required}")]
    AckTimeout {
        stage: FlashStage,
        recovery_required: bool,
    },
    #[error("device rejected {stage:?} with ACK payload {payload:02x?}; recovery_required={recovery_required}")]
    Rejected {
        stage: FlashStage,
        payload: Vec<u8>,
        recovery_required: bool,
    },
    #[error("invalid Enter-IAP identity: {detail}")]
    InvalidIapIdentity { detail: String },
    #[error("Enter-IAP device ID {actual:#010x} does not match authorized value {expected:#010x}")]
    IapDeviceMismatch { expected: u32, actual: u32 },
    #[error("Enter-IAP firmware ID {actual:#010x} does not match artifact value {expected:#010x}")]
    IapFirmwareMismatch { expected: u32, actual: u32 },
    #[error("invalid flash options: {0}")]
    InvalidOptions(&'static str),
    #[error("node id {0} is outside the addressable range 1..=127")]
    InvalidNode(u8),
}

impl FlashError {
    pub fn recovery_required(&self) -> bool {
        match self {
            Self::Cancelled {
                recovery_required, ..
            }
            | Self::Transport {
                recovery_required, ..
            }
            | Self::AckTimeout {
                recovery_required, ..
            }
            | Self::Rejected {
                recovery_required, ..
            } => *recovery_required,
            _ => false,
        }
    }
}

/// Run one complete update. `ready` must have been produced immediately before
/// this call from a complete identity reread on this exact `bus` instance.
/// Destructive requests are never automatically retransmitted.
pub async fn flash<F>(
    bus: &dyn CanBus,
    ready: ReadyToFlash,
    options: &FlashOptions,
    cancellation: &CancellationToken,
    mut on_event: F,
) -> Result<FlashOutcome, FlashError>
where
    F: FnMut(FlashEvent),
{
    validate_options(options)?;
    let node_id = ready.node_id();
    check_cancel(cancellation, FlashStage::Resetting, false)?;
    let rx_id = RX_BASE + node_id as u16;
    let mut rx = bus
        .subscribe(CanFilter::exact_standard(rx_id))
        .await
        .map_err(|error| FlashError::Transport {
            stage: FlashStage::Resetting,
            detail: format!("subscribing to exact ACK ID 0x{rx_id:03X}: {error}"),
            recovery_required: false,
        })?;
    let mut assembler = FrameAssembler::new();
    check_cancel(cancellation, FlashStage::Resetting, false)?;
    // Consume the short-lived identity capability immediately before the
    // first protocol-specific frame, after subscription setup has completed.
    let prepared = ready.into_prepared()?;
    let policy = match prepared.target().target().support() {
        crate::SupportPolicy::Enabled(policy) => policy,
        crate::SupportPolicy::Disabled { .. } => return Err(PolicyError::ProfileChanged.into()),
    };
    let artifact = prepared.artifact();

    let reset = Frame::request(node_id, FunctionCode::ResetRequest, Vec::new())
        .expect("validated node and protocol constant");
    let reset_ack = send_with_attempts(
        bus,
        rx.as_mut(),
        &mut assembler,
        &reset,
        AttemptPolicy {
            stage: FlashStage::Resetting,
            timeout: options.reset_timeout,
            attempts: options.reset_attempts,
            ambiguous_is_recovery: false,
        },
        options,
        &mut on_event,
    )
    .await?;
    require_success(reset_ack, FlashStage::Resetting, false)?;

    check_cancel(cancellation, FlashStage::EnteringBootloader, false)?;
    let enter_ack = knock_into_iap(
        bus,
        rx.as_mut(),
        &mut assembler,
        node_id,
        options,
        cancellation,
        &mut on_event,
    )
    .await?;
    write_artifact(
        bus,
        rx.as_mut(),
        &mut assembler,
        node_id,
        policy,
        artifact,
        enter_ack,
        options,
        cancellation,
        &mut on_event,
    )
    .await
}

/// Permission to write a device that cannot prove which device it is.
///
/// [`flash`] binds an update to one exact unit by requiring a complete CANopen
/// identity read from the very bus it is about to write. A bootloader implements
/// no CANopen at all, so a device whose application is gone cannot answer that
/// read and would otherwise be beyond repair with this crate.
///
/// This substitutes the caller's assertion of which profile the device follows,
/// and substitutes nothing else. The artifact must still satisfy that profile's
/// policy, and [`recover`] still refuses to write unless the identity the
/// bootloader reports on entering IAP agrees with both the policy and the
/// artifact. What is given up is the binding to one specific serial number: the
/// caller is stating it has identified the device some other way, and is
/// answerable for that.
#[derive(Debug, Clone)]
pub struct AssertedTarget {
    profile_id: String,
    policy: IapPolicy,
    artifact: ImgArtifact,
}

impl AssertedTarget {
    /// `target` must come from the caller's own registry, so that asserting a
    /// profile still cannot invent one.
    pub fn assert(target: &RegisteredTarget, artifact: ImgArtifact) -> Result<Self, PolicyError> {
        let SupportPolicy::Enabled(policy) = target.support() else {
            return Err(PolicyError::ProfileChanged);
        };
        policy.validate_artifact(&artifact)?;
        Ok(Self {
            profile_id: target.profile_id().to_owned(),
            policy: policy.clone(),
            artifact,
        })
    }

    pub fn profile_id(&self) -> &str {
        &self.profile_id
    }

    pub fn artifact(&self) -> &ImgArtifact {
        &self.artifact
    }
}

/// Write a device that answers only the IAP protocol.
///
/// Unlike [`flash`] this does not begin with a reset, because a device in this
/// state is not reliably reachable at all. Measured on hardware whose
/// application had been erased: it answers only inside the window that follows
/// its start, keeps acknowledging at the CAN level for a while after that window
/// closes, and eventually stops acknowledging altogether — by which point even a
/// reset cannot be transmitted to it. Callers are expected to have knocked the
/// device into IAP first, which [`scan_for_bootloaders`](crate::scan_for_bootloaders)
/// does and which is sticky once it takes.
///
/// The knock is repeated here anyway. It costs one frame against a device that
/// is already in IAP, and it is what produces the identity this refuses to write
/// without.
pub async fn recover<F>(
    bus: &dyn CanBus,
    node_id: u8,
    target: &AssertedTarget,
    options: &FlashOptions,
    cancellation: &CancellationToken,
    mut on_event: F,
) -> Result<FlashOutcome, FlashError>
where
    F: FnMut(FlashEvent),
{
    validate_options(options)?;
    if !(1..=127).contains(&node_id) {
        return Err(FlashError::InvalidNode(node_id));
    }
    check_cancel(cancellation, FlashStage::EnteringBootloader, false)?;
    let rx_id = RX_BASE + node_id as u16;
    let mut rx = bus
        .subscribe(CanFilter::exact_standard(rx_id))
        .await
        .map_err(|error| FlashError::Transport {
            stage: FlashStage::EnteringBootloader,
            detail: format!("subscribing to exact ACK ID 0x{rx_id:03X}: {error}"),
            recovery_required: false,
        })?;
    let mut assembler = FrameAssembler::new();

    let enter_ack = knock_into_iap(
        bus,
        rx.as_mut(),
        &mut assembler,
        node_id,
        options,
        cancellation,
        &mut on_event,
    )
    .await?;

    write_artifact(
        bus,
        rx.as_mut(),
        &mut assembler,
        node_id,
        &target.policy,
        &target.artifact,
        enter_ack,
        options,
        cancellation,
        &mut on_event,
    )
    .await
}

/// Everything from proving the bootloader's identity to the final verify.
///
/// Shared by `flash` and `recover`, which differ only in how the device was
/// brought to this point and in what authorized the write.
#[allow(clippy::too_many_arguments)]
async fn write_artifact<F>(
    bus: &dyn CanBus,
    rx: &mut dyn CanRx,
    assembler: &mut FrameAssembler,
    node_id: u8,
    policy: &IapPolicy,
    artifact: &ImgArtifact,
    enter_ack: Frame,
    options: &FlashOptions,
    cancellation: &CancellationToken,
    on_event: &mut F,
) -> Result<FlashOutcome, FlashError>
where
    F: FnMut(FlashEvent),
{
    let iap_identity =
        IapIdentity::parse(enter_ack.data()).map_err(|error| FlashError::InvalidIapIdentity {
            detail: error.to_string(),
        })?;
    if iap_identity.device_id() != policy.device_id() {
        return Err(FlashError::IapDeviceMismatch {
            expected: policy.device_id(),
            actual: iap_identity.device_id(),
        });
    }
    if iap_identity.firmware_id() != artifact.firmware_id() {
        return Err(FlashError::IapFirmwareMismatch {
            expected: artifact.firmware_id(),
            actual: iap_identity.firmware_id(),
        });
    }
    (on_event)(FlashEvent::Stage(FlashStage::IdentityVerified));
    (on_event)(FlashEvent::IapIdentity(iap_identity));

    check_cancel(cancellation, FlashStage::StartingDownload, false)?;
    let start = Frame::request(
        node_id,
        FunctionCode::StartDownloadRequest,
        artifact.tag().to_vec(),
    )
    .expect("IMG tag fits protocol payload");
    let start_ack = send_with_attempts(
        bus,
        &mut *rx,
        assembler,
        &start,
        AttemptPolicy {
            stage: FlashStage::StartingDownload,
            timeout: options.start_timeout,
            attempts: 1,
            ambiguous_is_recovery: true,
        },
        options,
        on_event,
    )
    .await?;
    // A definitive NAK means the device rejected metadata before erase.
    require_success(start_ack, FlashStage::StartingDownload, false)?;

    let total = artifact.bin_size();
    let mut written = 0usize;
    for (index, chunk) in artifact.bin().chunks(MAX_DATA_LEN).enumerate() {
        check_cancel(cancellation, FlashStage::Transferring, true)?;
        let function = if index % 2 == 0 {
            FunctionCode::SegmentARequest
        } else {
            FunctionCode::SegmentBRequest
        };
        let segment = Frame::request(node_id, function, chunk.to_vec())
            .expect("chunking respects protocol payload bound");
        let ack = send_with_attempts(
            bus,
            &mut *rx,
            assembler,
            &segment,
            AttemptPolicy {
                stage: FlashStage::Transferring,
                timeout: options.segment_timeout,
                attempts: 1,
                ambiguous_is_recovery: true,
            },
            options,
            on_event,
        )
        .await?;
        require_success(ack, FlashStage::Transferring, true)?;
        written += chunk.len();
        (on_event)(FlashEvent::Progress { written, total });
    }

    check_cancel(cancellation, FlashStage::Finalizing, true)?;
    let final_request = Frame::request(node_id, FunctionCode::FinalDownloadRequest, Vec::new())
        .expect("validated node and protocol constant");
    let final_ack = send_with_attempts(
        bus,
        &mut *rx,
        assembler,
        &final_request,
        AttemptPolicy {
            stage: FlashStage::Finalizing,
            timeout: options.final_timeout,
            attempts: 1,
            ambiguous_is_recovery: true,
        },
        options,
        on_event,
    )
    .await?;
    require_success(final_ack, FlashStage::Finalizing, true)?;

    check_cancel(cancellation, FlashStage::Verifying, true)?;
    let verify = Frame::request(
        node_id,
        FunctionCode::VerifyRequest,
        artifact.tag().to_vec(),
    )
    .expect("IMG tag fits protocol payload");
    let verify_ack = send_with_attempts(
        bus,
        &mut *rx,
        assembler,
        &verify,
        AttemptPolicy {
            stage: FlashStage::Verifying,
            timeout: options.verify_timeout,
            attempts: 1,
            ambiguous_is_recovery: true,
        },
        options,
        on_event,
    )
    .await?;
    require_success(verify_ack, FlashStage::Verifying, true)?;

    Ok(FlashOutcome { iap_identity })
}

fn validate_options(options: &FlashOptions) -> Result<(), FlashError> {
    if options.reset_attempts == 0 {
        return Err(FlashError::InvalidOptions(
            "attempt counts must be non-zero",
        ));
    }
    if options.tx_burst_frames == 0 {
        return Err(FlashError::InvalidOptions(
            "tx_burst_frames must be non-zero",
        ));
    }
    if options.knock_interval.is_zero() {
        return Err(FlashError::InvalidOptions(
            "knock_interval must be non-zero",
        ));
    }
    Ok(())
}

/// Enter IAP by knocking repeatedly instead of guessing one reboot delay.
///
/// The bootloader accepts `Enter_Iap_Req` only between finishing its own reset
/// and timing out into the application. A single delayed shot has to guess both
/// edges of that window; measured hardware opens it far later than the vendor's
/// documented 700 ms and closes it well before a 3 s ACK timeout would expire,
/// so the documented sequence cannot hit it at all. Knocking covers the window
/// wherever it happens to fall.
///
/// `Enter_Iap_Req` erases nothing and the device answers every knock once it is
/// in IAP, so repeating it is safe. Sends are best-effort on purpose: for most
/// of this window the target is still resetting, and when it is the only other
/// node on the bus nothing acknowledges the frame, which surfaces here as a
/// transport error that must not end the session.
async fn knock_into_iap<F>(
    bus: &dyn CanBus,
    rx: &mut dyn CanRx,
    assembler: &mut FrameAssembler,
    node_id: u8,
    options: &FlashOptions,
    cancellation: &CancellationToken,
    on_event: &mut F,
) -> Result<Frame, FlashError>
where
    F: FnMut(FlashEvent),
{
    let request = Frame::request(node_id, FunctionCode::EnterIapRequest, Vec::new())
        .expect("validated node and protocol constant");
    let stage = FlashStage::EnteringBootloader;
    assembler.reset();
    on_event(FlashEvent::Stage(stage));
    tokio::time::sleep(options.knock_start_delay).await;

    let deadline = tokio::time::Instant::now() + options.knock_window;
    loop {
        check_cancel(cancellation, stage, false)?;
        let _ = send_frame(bus, &request, stage, false, options).await;

        let now = tokio::time::Instant::now();
        let listen = options
            .knock_interval
            .min(deadline.saturating_duration_since(now));
        match receive_expected(
            rx,
            assembler,
            node_id,
            FunctionCode::EnterIapAck,
            listen,
            stage,
        )
        .await
        {
            Ok(frame) => return Ok(frame),
            Err(FlashError::AckTimeout { .. }) => {}
            Err(error) => return Err(error),
        }

        if tokio::time::Instant::now() >= deadline {
            return Err(FlashError::AckTimeout {
                stage,
                recovery_required: false,
            });
        }
    }
}

fn check_cancel(
    cancellation: &CancellationToken,
    stage: FlashStage,
    recovery_required: bool,
) -> Result<(), FlashError> {
    if cancellation.is_cancelled() {
        return Err(FlashError::Cancelled {
            stage,
            recovery_required,
        });
    }
    Ok(())
}

#[derive(Debug, Clone, Copy)]
struct AttemptPolicy {
    stage: FlashStage,
    timeout: Duration,
    attempts: u8,
    ambiguous_is_recovery: bool,
}

async fn send_with_attempts<F>(
    bus: &dyn CanBus,
    rx: &mut dyn CanRx,
    assembler: &mut FrameAssembler,
    request: &Frame,
    policy: AttemptPolicy,
    options: &FlashOptions,
    on_event: &mut F,
) -> Result<Frame, FlashError>
where
    F: FnMut(FlashEvent),
{
    let expected = request
        .function()
        .expected_ack()
        .expect("callers pass requests");
    for attempt in 0..policy.attempts {
        assembler.reset();
        send_frame(
            bus,
            request,
            policy.stage,
            policy.ambiguous_is_recovery,
            options,
        )
        .await?;
        on_event(FlashEvent::Stage(policy.stage));
        match receive_expected(
            rx,
            assembler,
            request.node_id(),
            expected,
            policy.timeout,
            policy.stage,
        )
        .await
        {
            Ok(frame) => return Ok(frame),
            Err(FlashError::AckTimeout { .. }) if attempt + 1 < policy.attempts => continue,
            Err(error) => return Err(with_recovery(error, policy.ambiguous_is_recovery)),
        }
    }
    unreachable!("attempt count is validated as non-zero")
}

async fn send_frame(
    bus: &dyn CanBus,
    request: &Frame,
    stage: FlashStage,
    recovery_required: bool,
    options: &FlashOptions,
) -> Result<(), FlashError> {
    let tx_id = TX_BASE + request.node_id() as u16;
    for (index, chunk) in request.encode().chunks(8).enumerate() {
        let can_frame =
            CanFrame::new_data(tx_id, chunk).map_err(|error| FlashError::Transport {
                stage,
                detail: format!("constructing Classic CAN frame: {error}"),
                recovery_required,
            })?;
        bus.send(can_frame)
            .await
            .map_err(|error| FlashError::Transport {
                stage,
                detail: format!(
                    "sending COBS chunk {} (the request outcome is ambiguous): {error}",
                    index + 1
                ),
                recovery_required,
            })?;
        if (index + 1) % options.tx_burst_frames == 0 {
            tokio::time::sleep(options.tx_burst_pause).await;
        }
    }
    Ok(())
}

async fn receive_expected(
    rx: &mut dyn CanRx,
    assembler: &mut FrameAssembler,
    node_id: u8,
    expected: FunctionCode,
    timeout: Duration,
    stage: FlashStage,
) -> Result<Frame, FlashError> {
    let deadline = tokio::time::Instant::now() + timeout;
    loop {
        let now = tokio::time::Instant::now();
        if now >= deadline {
            return Err(FlashError::AckTimeout {
                stage,
                recovery_required: false,
            });
        }
        let frame = match tokio::time::timeout(deadline - now, rx.recv()).await {
            Ok(Ok(frame)) => frame,
            Ok(Err(error)) => {
                return Err(FlashError::Transport {
                    stage,
                    detail: format!("receiving ACK: {error}"),
                    recovery_required: false,
                })
            }
            Err(_) => {
                return Err(FlashError::AckTimeout {
                    stage,
                    recovery_required: false,
                })
            }
        };
        if frame.is_fd() || frame.is_remote() {
            continue;
        }
        let CanId::Standard(id) = frame.id() else {
            continue;
        };
        if id != RX_BASE + node_id as u16 {
            continue;
        }
        for decoded in assembler.feed(frame.data()) {
            let Ok(decoded) = decoded else {
                continue;
            };
            if decoded.node_id() == node_id && decoded.function() == expected {
                return Ok(decoded);
            }
        }
    }
}

fn require_success(
    ack: Frame,
    stage: FlashStage,
    recovery_required: bool,
) -> Result<(), FlashError> {
    if ack.data() == [1] {
        Ok(())
    } else {
        Err(FlashError::Rejected {
            stage,
            payload: ack.data().to_vec(),
            recovery_required,
        })
    }
}

fn with_recovery(error: FlashError, recovery_required: bool) -> FlashError {
    match error {
        FlashError::Transport { stage, detail, .. } => FlashError::Transport {
            stage,
            detail,
            recovery_required,
        },
        FlashError::AckTimeout { stage, .. } => FlashError::AckTimeout {
            stage,
            recovery_required,
        },
        other => other,
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use async_trait::async_trait;
    use can_transport::{CanCapabilities, CanIoError};
    use tokio::sync::mpsc;

    use crate::artifact::tests::make_image;
    use crate::{
        CanopenIdentity, IapPolicy, ImgArtifact, ImgLimits, PreparedUpgrade, RegisteredTarget,
        TargetRegistry,
    };

    use super::*;

    struct FakeBus {
        request_assembler: Mutex<FrameAssembler>,
        requests: Mutex<Vec<FunctionCode>>,
        receiver_tx: Mutex<Option<mpsc::UnboundedSender<CanFrame>>>,
        drop_ack: Option<FunctionCode>,
        nak: Option<FunctionCode>,
        /// Enter-IAP knocks to swallow before the bootloader window "opens".
        deaf_knocks: Mutex<usize>,
    }

    impl FakeBus {
        fn new(drop_ack: Option<FunctionCode>, nak: Option<FunctionCode>) -> Self {
            Self {
                request_assembler: Mutex::new(FrameAssembler::new()),
                requests: Mutex::new(Vec::new()),
                receiver_tx: Mutex::new(None),
                drop_ack,
                nak,
                deaf_knocks: Mutex::new(0),
            }
        }

        fn deaf_for(self, knocks: usize) -> Self {
            *self.deaf_knocks.lock().unwrap() = knocks;
            self
        }

        fn requests(&self) -> Vec<FunctionCode> {
            self.requests.lock().unwrap().clone()
        }
    }

    #[async_trait]
    impl CanBus for FakeBus {
        async fn send(&self, frame: CanFrame) -> Result<(), CanIoError> {
            let decoded = self.request_assembler.lock().unwrap().feed(frame.data());
            for request in decoded {
                let request = request.map_err(CanIoError::backend)?;
                self.requests.lock().unwrap().push(request.function());
                if self.drop_ack == Some(request.function()) {
                    continue;
                }
                if request.function() == FunctionCode::EnterIapRequest {
                    let mut deaf = self.deaf_knocks.lock().unwrap();
                    if *deaf > 0 {
                        *deaf -= 1;
                        continue;
                    }
                }
                let function = request.function().expected_ack().unwrap();
                let data = if function == FunctionCode::EnterIapAck {
                    [
                        0xAAAA_0001u32.to_le_bytes(),
                        0x2025_1025u32.to_le_bytes(),
                        8u32.to_le_bytes(),
                    ]
                    .concat()
                } else if self.nak == Some(request.function()) {
                    vec![0]
                } else {
                    vec![1]
                };
                let ack = Frame::new(request.node_id(), function, data).unwrap();
                let sender = self.receiver_tx.lock().unwrap().clone().unwrap();
                for chunk in ack.encode().chunks(8) {
                    sender
                        .send(CanFrame::new_data(
                            RX_BASE + request.node_id() as u16,
                            chunk,
                        )?)
                        .map_err(|_| CanIoError::Disconnected)?;
                }
            }
            Ok(())
        }

        async fn subscribe(&self, filter: CanFilter) -> Result<Box<dyn CanRx>, CanIoError> {
            assert_eq!(filter, CanFilter::exact_standard(RX_BASE + 1));
            let (sender, receiver) = mpsc::unbounded_channel();
            *self.receiver_tx.lock().unwrap() = Some(sender);
            Ok(Box::new(FakeRx { receiver }))
        }

        fn capabilities(&self) -> CanCapabilities {
            CanCapabilities {
                fd: false,
                max_dlen: 8,
            }
        }
    }

    struct FakeRx {
        receiver: mpsc::UnboundedReceiver<CanFrame>,
    }

    #[async_trait]
    impl CanRx for FakeRx {
        async fn recv(&mut self) -> Result<CanFrame, CanIoError> {
            self.receiver.recv().await.ok_or(CanIoError::Disconnected)
        }

        fn try_recv(&mut self) -> Result<Option<CanFrame>, CanIoError> {
            match self.receiver.try_recv() {
                Ok(frame) => Ok(Some(frame)),
                Err(mpsc::error::TryRecvError::Empty) => Ok(None),
                Err(mpsc::error::TryRecvError::Disconnected) => Err(CanIoError::Disconnected),
            }
        }
    }

    fn ready(bin_len: usize) -> ReadyToFlash {
        let identity = CanopenIdentity::new(1, 0x4859_444C, 0xAAAA_0001, 7, 123).unwrap();
        let registry = TargetRegistry::new(vec![RegisteredTarget::enabled(
            "compatible-motor-4310-v1",
            0x4859_444C,
            0xAAAA_0001,
            IapPolicy::new(0xAAAA_0001, vec![0x2025_1025], 0x1000_C000, true).unwrap(),
        )
        .unwrap()])
        .unwrap();
        let target = registry.authorize(identity).unwrap();
        let bytes = make_image(0xAAAA_0001, 0x2025_1025, 0x1000_C000, bin_len, true);
        let artifact = ImgArtifact::parse(&bytes, ImgLimits::default()).unwrap();
        PreparedUpgrade::bind(target, artifact)
            .unwrap()
            .revalidate(identity, &registry)
            .unwrap()
    }

    fn fast_options() -> FlashOptions {
        FlashOptions {
            knock_start_delay: Duration::ZERO,
            knock_interval: Duration::from_millis(5),
            knock_window: Duration::from_millis(50),
            reset_timeout: Duration::from_millis(5),
            start_timeout: Duration::from_millis(5),
            segment_timeout: Duration::from_millis(5),
            final_timeout: Duration::from_millis(5),
            verify_timeout: Duration::from_millis(5),
            reset_attempts: 1,
            tx_burst_frames: 15,
            tx_burst_pause: Duration::ZERO,
        }
    }

    #[tokio::test]
    async fn fake_device_runs_the_exact_six_stage_flow() {
        let bus = FakeBus::new(None, None);
        let mut events = Vec::new();
        let outcome = flash(
            &bus,
            ready(300),
            &fast_options(),
            &CancellationToken::new(),
            |event| events.push(event),
        )
        .await
        .unwrap();
        assert_eq!(outcome.iap_identity().current_version(), 8);
        assert_eq!(
            bus.requests(),
            vec![
                FunctionCode::ResetRequest,
                FunctionCode::EnterIapRequest,
                FunctionCode::StartDownloadRequest,
                FunctionCode::SegmentARequest,
                FunctionCode::SegmentBRequest,
                FunctionCode::FinalDownloadRequest,
                FunctionCode::VerifyRequest,
            ]
        );
        assert!(events.contains(&FlashEvent::Progress {
            written: 300,
            total: 300,
        }));
    }

    #[tokio::test]
    async fn enter_iap_keeps_knocking_until_the_bootloader_window_opens() {
        // A device that ignores the first five knocks still gets entered: the
        // single documented shot at a fixed reboot delay would have missed it.
        let bus = FakeBus::new(None, None).deaf_for(5);
        flash(
            &bus,
            ready(16),
            &fast_options(),
            &CancellationToken::new(),
            |_| {},
        )
        .await
        .unwrap();
        let knocks = bus
            .requests()
            .iter()
            .filter(|function| **function == FunctionCode::EnterIapRequest)
            .count();
        assert_eq!(knocks, 6);
    }

    #[tokio::test]
    async fn a_window_that_never_opens_fails_before_anything_destructive() {
        let bus = FakeBus::new(None, None).deaf_for(usize::MAX);
        let error = flash(
            &bus,
            ready(16),
            &fast_options(),
            &CancellationToken::new(),
            |_| {},
        )
        .await
        .unwrap_err();
        assert!(matches!(
            error,
            FlashError::AckTimeout {
                stage: FlashStage::EnteringBootloader,
                recovery_required: false,
            }
        ));
        assert!(bus.requests().iter().all(|function| matches!(
            function,
            FunctionCode::ResetRequest | FunctionCode::EnterIapRequest
        )));
    }

    #[tokio::test]
    async fn destructive_timeout_is_not_retried() {
        let bus = FakeBus::new(Some(FunctionCode::StartDownloadRequest), None);
        let error = flash(
            &bus,
            ready(16),
            &fast_options(),
            &CancellationToken::new(),
            |_| {},
        )
        .await
        .unwrap_err();
        assert!(matches!(
            error,
            FlashError::AckTimeout {
                stage: FlashStage::StartingDownload,
                recovery_required: true,
            }
        ));
        assert_eq!(
            bus.requests()
                .iter()
                .filter(|function| **function == FunctionCode::StartDownloadRequest)
                .count(),
            1
        );
    }

    #[tokio::test]
    async fn definitive_start_nak_is_before_erase_rejection() {
        let bus = FakeBus::new(None, Some(FunctionCode::StartDownloadRequest));
        let error = flash(
            &bus,
            ready(16),
            &fast_options(),
            &CancellationToken::new(),
            |_| {},
        )
        .await
        .unwrap_err();
        assert!(matches!(
            error,
            FlashError::Rejected {
                stage: FlashStage::StartingDownload,
                recovery_required: false,
                ..
            }
        ));
    }

    #[tokio::test]
    async fn cancellation_before_reset_sends_nothing() {
        let bus = FakeBus::new(None, None);
        let cancellation = CancellationToken::new();
        cancellation.cancel();
        let error = flash(&bus, ready(16), &fast_options(), &cancellation, |_| {})
            .await
            .unwrap_err();
        assert!(matches!(
            error,
            FlashError::Cancelled {
                stage: FlashStage::Resetting,
                recovery_required: false,
            }
        ));
        assert!(bus.requests().is_empty());
    }

    fn motor_4310() -> RegisteredTarget {
        RegisteredTarget::enabled(
            "compatible-motor-4310-v1",
            0x4859_444C,
            0xAAAA_0001,
            IapPolicy::new(0xAAAA_0001, vec![0x2025_1025], 0x1000_C000, true).unwrap(),
        )
        .unwrap()
    }

    fn asserted(bin_len: usize) -> AssertedTarget {
        let bytes = make_image(0xAAAA_0001, 0x2025_1025, 0x1000_C000, bin_len, true);
        let artifact = ImgArtifact::parse(&bytes, ImgLimits::default()).unwrap();
        AssertedTarget::assert(&motor_4310(), artifact).unwrap()
    }

    #[tokio::test]
    async fn recovery_writes_without_a_reset() {
        // A device that has stopped acknowledging cannot be sent a reset at all,
        // so recovery has to start from the knock.
        let bus = FakeBus::new(None, None);
        recover(
            &bus,
            1,
            &asserted(300),
            &fast_options(),
            &CancellationToken::new(),
            |_| {},
        )
        .await
        .unwrap();
        assert_eq!(
            bus.requests(),
            vec![
                FunctionCode::EnterIapRequest,
                FunctionCode::StartDownloadRequest,
                FunctionCode::SegmentARequest,
                FunctionCode::SegmentBRequest,
                FunctionCode::FinalDownloadRequest,
                FunctionCode::VerifyRequest,
            ]
        );
    }

    #[tokio::test]
    async fn asserting_a_profile_does_not_excuse_the_artifact() {
        // Naming the profile replaces the identity read and nothing else.
        let bytes = make_image(0xAAAA_0001, 0x2025_1209, 0x1000_C000, 16, true);
        let artifact = ImgArtifact::parse(&bytes, ImgLimits::default()).unwrap();
        assert!(AssertedTarget::assert(&motor_4310(), artifact).is_err());
    }

    #[tokio::test]
    async fn a_disabled_profile_cannot_be_asserted() {
        let target = RegisteredTarget::disabled(
            "compatible-motor-4360-v1",
            0x4859_444C,
            0xAAAA_0005,
            "not qualified",
        )
        .unwrap();
        let bytes = make_image(0xAAAA_0001, 0x2025_1025, 0x1000_C000, 16, true);
        let artifact = ImgArtifact::parse(&bytes, ImgLimits::default()).unwrap();
        assert!(matches!(
            AssertedTarget::assert(&target, artifact),
            Err(PolicyError::ProfileChanged)
        ));
    }

    #[tokio::test]
    async fn recovery_still_refuses_an_unaddressable_node() {
        let bus = FakeBus::new(None, None);
        let error = recover(
            &bus,
            0,
            &asserted(16),
            &fast_options(),
            &CancellationToken::new(),
            |_| {},
        )
        .await
        .unwrap_err();
        assert!(matches!(error, FlashError::InvalidNode(0)));
        assert!(bus.requests().is_empty());
    }
}
