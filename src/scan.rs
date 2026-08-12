//! Find bootloaders that cannot announce themselves.
//!
//! A device whose application was erased by an interrupted update stays in the
//! bootloader permanently, and the bootloader speaks only this protocol: no
//! CANopen, no heartbeat, no 0x1018, and nothing at all until it is addressed.
//! Recovering one means knowing its node ID, which is exactly what is lost when
//! an update goes wrong. Probing every address is the only way to find it.

use std::collections::HashMap;
use std::time::Duration;

use can_transport::{CanBus, CanFilter, CanId};
use thiserror::Error;
use tokio::time::Instant;

use crate::engine::{RX_BASE, TX_BASE};
use crate::{Frame, FrameAssembler, FunctionCode, IapIdentity};

/// Matches 0x680..=0x6FF, the answer ID of every node, and nothing else.
const ACK_ID_MASK: u16 = 0x780;

/// Probes to send before pausing, matching the flash path's transmit pacing.
const PROBE_BURST: usize = 4;
const PROBE_PAUSE: Duration = Duration::from_millis(1);
/// How long one probe may spend waiting for room in the transmit queue. A full
/// sweep has to fit inside a bootloader window, so this is far shorter than the
/// transport's own back-pressure tolerance.
const PROBE_SEND_TIMEOUT: Duration = Duration::from_millis(5);

#[derive(Debug, Error)]
pub enum ScanError {
    #[error("node id {0} is outside the addressable range 1..=127")]
    InvalidNode(u8),
    #[error("CAN transport failed during the scan: {0}")]
    Transport(String),
}

/// One node that answered a scan probe.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BootloaderHit {
    node_id: u8,
    identity: Option<IapIdentity>,
    payload: Vec<u8>,
}

impl BootloaderHit {
    pub fn node_id(&self) -> u8 {
        self.node_id
    }

    /// The reported identity, or `None` when the bootloader answered with the
    /// all-0xFF payload. That answer is not a fault: it is how a device reports
    /// that it holds no valid application, which is precisely the state an
    /// interrupted update leaves behind.
    pub fn identity(&self) -> Option<IapIdentity> {
        self.identity
    }

    /// Whether the device reported a usable application identity.
    pub fn has_application(&self) -> bool {
        self.identity.is_some()
    }

    /// The raw Enter-IAP payload, kept so callers can show what an
    /// unparseable answer actually contained.
    pub fn payload(&self) -> &[u8] {
        &self.payload
    }
}

/// What one sweep observed.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ScanReport {
    hits: Vec<BootloaderHit>,
    probed: usize,
    unsent: usize,
}

impl ScanReport {
    pub fn hits(&self) -> &[BootloaderHit] {
        &self.hits
    }

    pub fn into_hits(self) -> Vec<BootloaderHit> {
        self.hits
    }

    pub fn probed(&self) -> usize {
        self.probed
    }

    /// Probes that could not even be handed to the bus.
    pub fn unsent(&self) -> usize {
        self.unsent
    }

    /// Not one probe reached the wire.
    ///
    /// A CAN controller needs some other node to acknowledge each frame, so a
    /// bus whose only device is unpowered or hung cannot transmit at all: the
    /// first frame retransmits forever and the queue never drains. Reporting
    /// that as "no bootloader answered" would be misleading — nothing was
    /// asked.
    pub fn nothing_acknowledging(&self) -> bool {
        self.probed > 0 && self.unsent == self.probed
    }
}

/// Probe `nodes` and collect every bootloader that answers within `listen`.
///
/// `Enter_Iap_Req` is the only request safe to probe with: it erases nothing,
/// and a running application ignores it, so a healthy motor is not disturbed.
///
/// It is not entirely free of effect, though. A device that happens to be
/// inside its post-reset bootloader window when the probe arrives will enter
/// IAP and stay there instead of starting its application. That is harmless —
/// a power cycle starts the application again — but it means a scan is a
/// recovery tool, not something to run against motors that are mid-boot.
pub async fn scan_for_bootloaders(
    bus: &dyn CanBus,
    nodes: impl IntoIterator<Item = u8>,
    listen: Duration,
) -> Result<ScanReport, ScanError> {
    let probes = nodes
        .into_iter()
        .map(|node_id| {
            Frame::request(node_id, FunctionCode::EnterIapRequest, Vec::new())
                .map_err(|_| ScanError::InvalidNode(node_id))
                .map(|frame| (node_id, frame))
        })
        .collect::<Result<Vec<_>, _>>()?;

    // Subscribe before probing so an answer to the first probe cannot be missed.
    let mut rx = bus
        .subscribe(CanFilter::standard(RX_BASE, ACK_ID_MASK))
        .await
        .map_err(|error| {
            ScanError::Transport(format!("subscribing to the answer range: {error}"))
        })?;

    let mut unsent = 0usize;
    for (index, (node_id, probe)) in probes.iter().enumerate() {
        let tx_id = TX_BASE + *node_id as u16;
        for chunk in probe.encode().chunks(8) {
            // Best-effort, and deliberately impatient. Most addresses have
            // nothing behind them, and the device being hunted may be the only
            // one on the bus, in which case nothing acknowledges and the
            // transport's back-pressure wait would stall this sweep for far
            // longer than a bootloader window lasts. Give up on a frame quickly
            // and rely on sweeping again.
            let Ok(frame) = can_transport::CanFrame::new_data(tx_id, chunk) else {
                continue;
            };
            match tokio::time::timeout(PROBE_SEND_TIMEOUT, bus.send(frame)).await {
                Ok(Ok(())) => {}
                Ok(Err(_)) | Err(_) => unsent += 1,
            }
        }
        if (index + 1) % PROBE_BURST == 0 {
            tokio::time::sleep(PROBE_PAUSE).await;
        }
    }

    // One assembler per node: answers from different nodes interleave on the
    // bus, and a shared assembler would splice them into corrupt frames.
    let mut assemblers: HashMap<u8, FrameAssembler> = HashMap::new();
    let mut hits: Vec<BootloaderHit> = Vec::new();
    let deadline = Instant::now() + listen;
    loop {
        let now = Instant::now();
        if now >= deadline {
            break;
        }
        let frame = match tokio::time::timeout(deadline - now, rx.recv()).await {
            Ok(Ok(frame)) => frame,
            Ok(Err(error)) => return Err(ScanError::Transport(format!("receiving: {error}"))),
            Err(_) => break,
        };
        if frame.is_fd() || frame.is_remote() {
            continue;
        }
        let CanId::Standard(id) = frame.id() else {
            continue;
        };
        let Some(node_id) = id
            .checked_sub(RX_BASE)
            .filter(|node| (1..=127).contains(node))
            .map(|node| node as u8)
        else {
            continue;
        };
        let assembler = assemblers.entry(node_id).or_default();
        for decoded in assembler.feed(frame.data()) {
            let Ok(decoded) = decoded else {
                continue;
            };
            if decoded.node_id() != node_id || decoded.function() != FunctionCode::EnterIapAck {
                continue;
            }
            if hits.iter().any(|hit| hit.node_id == node_id) {
                continue;
            }
            hits.push(BootloaderHit {
                node_id,
                identity: IapIdentity::parse(decoded.data()).ok(),
                payload: decoded.data().to_vec(),
            });
        }
    }

    hits.sort_by_key(|hit| hit.node_id);
    Ok(ScanReport {
        hits,
        probed: probes.len(),
        unsent,
    })
}

/// Probe every addressable node. Node 0 is the reserved broadcast address.
pub async fn scan_all(bus: &dyn CanBus, listen: Duration) -> Result<ScanReport, ScanError> {
    scan_for_bootloaders(bus, 1..=127, listen).await
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use async_trait::async_trait;
    use can_transport::{CanCapabilities, CanFrame, CanIoError, CanRx};
    use tokio::sync::mpsc;

    use super::*;

    /// Answers on behalf of the node IDs it was given, with the payload each
    /// should report.
    struct FakeBus {
        answering: Vec<(u8, Vec<u8>)>,
        request_assemblers: Mutex<HashMap<u16, FrameAssembler>>,
        receiver_tx: Mutex<Option<mpsc::UnboundedSender<CanFrame>>>,
    }

    impl FakeBus {
        fn new(answering: Vec<(u8, Vec<u8>)>) -> Self {
            Self {
                answering,
                request_assemblers: Mutex::new(HashMap::new()),
                receiver_tx: Mutex::new(None),
            }
        }
    }

    #[async_trait]
    impl CanBus for FakeBus {
        async fn send(&self, frame: CanFrame) -> Result<(), CanIoError> {
            let CanId::Standard(id) = frame.id() else {
                return Ok(());
            };
            let decoded = {
                let mut assemblers = self.request_assemblers.lock().unwrap();
                let assembler = assemblers.entry(id).or_default();
                assembler.feed(frame.data())
            };
            for request in decoded.into_iter().flatten() {
                let Some((_, payload)) = self
                    .answering
                    .iter()
                    .find(|(node, _)| *node == request.node_id())
                else {
                    continue;
                };
                let ack = Frame::new(
                    request.node_id(),
                    FunctionCode::EnterIapAck,
                    payload.clone(),
                )
                .unwrap();
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
            assert_eq!(filter, CanFilter::standard(RX_BASE, ACK_ID_MASK));
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
            Ok(self.receiver.try_recv().ok())
        }
    }

    fn identity_payload(device_id: u32, firmware_id: u32, version: u32) -> Vec<u8> {
        [
            device_id.to_le_bytes(),
            firmware_id.to_le_bytes(),
            version.to_le_bytes(),
        ]
        .concat()
    }

    #[tokio::test]
    async fn finds_every_answering_node_and_ignores_silent_addresses() {
        let bus = FakeBus::new(vec![
            (7, identity_payload(0xAAAA_0001, 0x2025_1025, 10)),
            (42, identity_payload(0xAAAA_0002, 0x2025_1209, 3)),
        ]);
        let report = scan_all(&bus, Duration::from_millis(50)).await.unwrap();
        let hits = report.hits();
        assert_eq!(
            hits.iter().map(|hit| hit.node_id()).collect::<Vec<_>>(),
            vec![7, 42]
        );
        assert_eq!(hits[0].identity().unwrap().current_version(), 10);
        assert_eq!(hits[1].identity().unwrap().device_id(), 0xAAAA_0002);
        assert!(hits.iter().all(BootloaderHit::has_application));
        assert_eq!(report.probed(), 127);
        assert_eq!(report.unsent(), 0);
        assert!(!report.nothing_acknowledging());
    }

    #[tokio::test]
    async fn an_erased_device_is_reported_rather_than_dropped() {
        // All-0xFF is how a bootloader says it holds no application. That node
        // is the one a recovery scan exists to find, so it must not be skipped
        // just because the payload does not parse as an identity.
        let bus = FakeBus::new(vec![(9, vec![0xFF; 12])]);
        let report = scan_all(&bus, Duration::from_millis(50)).await.unwrap();
        let hits = report.hits();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].node_id(), 9);
        assert!(!hits[0].has_application());
        assert_eq!(hits[0].payload(), [0xFF; 12]);
    }

    #[tokio::test]
    async fn a_silent_bus_yields_nothing_rather_than_failing() {
        let bus = FakeBus::new(Vec::new());
        let report = scan_all(&bus, Duration::from_millis(20)).await.unwrap();
        assert!(report.hits().is_empty());
        // Probes went out and simply went unanswered, which is a different
        // condition from not being able to transmit at all.
        assert!(!report.nothing_acknowledging());
    }

    #[tokio::test]
    async fn a_bus_that_cannot_transmit_is_not_reported_as_no_answer() {
        // An unpowered or hung sole device leaves the controller with nothing
        // to acknowledge its frames, so nothing is ever asked. Saying "no
        // bootloader answered" would send someone hunting the wrong fault.
        // The subscription itself stays healthy on a real dead bus, so hold the
        // sender: dropping it would make recv fail for a reason the hardware
        // never produces.
        #[derive(Default)]
        struct DeadBus {
            keepalive: Mutex<Option<mpsc::UnboundedSender<CanFrame>>>,
        }

        #[async_trait]
        impl CanBus for DeadBus {
            async fn send(&self, _frame: CanFrame) -> Result<(), CanIoError> {
                Err(CanIoError::Disconnected)
            }

            async fn subscribe(&self, _filter: CanFilter) -> Result<Box<dyn CanRx>, CanIoError> {
                let (sender, receiver) = mpsc::unbounded_channel();
                *self.keepalive.lock().unwrap() = Some(sender);
                Ok(Box::new(FakeRx { receiver }))
            }

            fn capabilities(&self) -> CanCapabilities {
                CanCapabilities {
                    fd: false,
                    max_dlen: 8,
                }
            }
        }

        let report = scan_all(&DeadBus::default(), Duration::from_millis(10))
            .await
            .unwrap();
        assert!(report.hits().is_empty());
        assert_eq!(report.probed(), 127);
        assert_eq!(report.unsent(), 127);
        assert!(report.nothing_acknowledging());
    }

    #[tokio::test]
    async fn the_broadcast_address_is_never_probed() {
        let bus = FakeBus::new(Vec::new());
        let error = scan_for_bootloaders(&bus, [0], Duration::from_millis(10))
            .await
            .unwrap_err();
        assert!(matches!(error, ScanError::InvalidNode(0)));
    }
}
