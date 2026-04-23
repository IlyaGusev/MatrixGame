//! Port of `SOrder` + `CMatrixRobotAI::m_OrdersList[MAX_ORDERS]`
//! (MatrixRobot.hpp:123-231). The C++ stores orders in a ring-like
//! array that's MoveMemory-shuffled on insert/remove; we use a
//! fixed-capacity `Vec<Order>` with manual insert-at-top + remove-
//! at-index for the same push-to-top semantics.
//!
//! Only the move / stop orders are ported here — capture / fire /
//! stop-fire land with combat. The data fields stay 1:1 so those
//! phases can extend without migration.
//!
//! `Params` carry:
//!   ROT_MOVE_TO   : p1 = destination move-cell X
//!                   p2 = destination move-cell Y
//!   ROT_MOVE_TO_BACK: same as MOVE_TO, drives in reverse
//!   ROT_MOVE_RETURN: p1,p2 = return-to move-cell X,Y
//!   ROT_STOP_MOVE : unused
//!   ROT_FIRE      : p1..3 = world-space target; p4 = type flag
//!   ROT_STOP_*    : unused

/// Port of `OrderType` (MatrixRobot.hpp:123-136). `Empty` stands in
/// for `ROT_EMPTY_ORDER`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum OrderType {
    Empty,
    MoveTo,
    MoveToBack,
    MoveReturn,
    StopMove,
    Fire,
    StopFire,
    CaptureFactory,
    StopCapture,
}

/// Port of `OrderPhase` (MatrixRobot.hpp:138-151).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum OrderPhase {
    Empty,
    WaitingForParams,
    Moving,
    Firing,
    CaptureMoving,
    CaptureInPosition,
    CaptureSettingUp,
    Capturing,
    GetingLost,
}

#[derive(Debug, Clone, Copy)]
pub struct Order {
    pub ty: OrderType,
    pub phase: OrderPhase,
    pub p1: f32,
    pub p2: f32,
    pub p3: f32,
    pub p4: i32,
}

impl Default for Order {
    fn default() -> Self {
        Self {
            ty: OrderType::Empty,
            phase: OrderPhase::Empty,
            p1: 0.0,
            p2: 0.0,
            p3: 0.0,
            p4: 0,
        }
    }
}

impl Order {
    /// Port of `SOrder::SetOrder(type, p1, p2, p3, p4)` (MatrixRobot.hpp:
    /// 198). Also resets the phase to `Empty` — the C++ calls
    /// `Reset()` first, which memsets the struct, so the phase
    /// starts fresh.
    pub fn set(ty: OrderType, p1: f32, p2: f32, p3: f32, p4: i32) -> Self {
        Self {
            ty,
            phase: OrderPhase::Empty,
            p1,
            p2,
            p3,
            p4,
        }
    }
}

/// `MAX_ORDERS` from MatrixRobot.hpp:33.
pub const MAX_ORDERS: usize = 5;

/// Port of `m_OrdersList[MAX_ORDERS]` + `m_OrdersInPool`. The C++
/// `AllocPlaceForOrderOnTop` shifts every existing order +1 and
/// returns `&m_OrdersList[0]` — so the head of the list is always
/// the most recently-added order. `OrderList::push_top` does that.
#[derive(Debug, Default, Clone)]
pub struct OrderList {
    orders: Vec<Order>,
}

impl OrderList {
    pub fn new() -> Self {
        Self {
            orders: Vec::with_capacity(MAX_ORDERS),
        }
    }

    pub fn len(&self) -> usize {
        self.orders.len()
    }
    pub fn is_empty(&self) -> bool {
        self.orders.is_empty()
    }
    pub fn is_full(&self) -> bool {
        self.orders.len() >= MAX_ORDERS
    }

    pub fn top(&self) -> Option<&Order> {
        self.orders.first()
    }
    pub fn top_mut(&mut self) -> Option<&mut Order> {
        self.orders.first_mut()
    }

    pub fn iter(&self) -> std::slice::Iter<'_, Order> {
        self.orders.iter()
    }
    pub fn iter_mut(&mut self) -> std::slice::IterMut<'_, Order> {
        self.orders.iter_mut()
    }

    /// Port of `CMatrixRobotAI::AllocPlaceForOrderOnTop`
    /// (MatrixRobot.cpp:4554-4570). Rejects if pool full.
    pub fn push_top(&mut self, order: Order) -> bool {
        if self.is_full() {
            return false;
        }
        self.orders.insert(0, order);
        true
    }

    /// Port of `CMatrixRobotAI::RemoveOrderFromTop`
    /// (MatrixRobot.cpp equivalent around :4612). Pops the head.
    pub fn pop_top(&mut self) -> Option<Order> {
        if self.orders.is_empty() {
            None
        } else {
            Some(self.orders.remove(0))
        }
    }

    /// Port of `CMatrixRobotAI::RemoveOrder(OrderType)`
    /// (MatrixRobot.cpp:4591-4602). Removes every order matching `ty`.
    pub fn remove_type(&mut self, ty: OrderType) {
        self.orders.retain(|o| o.ty != ty);
    }

    /// Port of `CMatrixRobotAI::FindOrderLikeThat(OrderType)`
    /// (MatrixRobot.cpp:4983-4991).
    pub fn has(&self, ty: OrderType) -> bool {
        self.orders.iter().any(|o| o.ty == ty)
    }

    /// Port of `CMatrixRobotAI::FindOrderLikeThat(OrderType, OrderPhase)`
    /// (MatrixRobot.cpp:4993-4999).
    pub fn has_with_phase(&self, ty: OrderType, phase: OrderPhase) -> bool {
        self.orders.iter().any(|o| o.ty == ty && o.phase == phase)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn push_top_limits_to_max_orders() {
        let mut ol = OrderList::new();
        for _ in 0..MAX_ORDERS {
            assert!(ol.push_top(Order::set(OrderType::MoveTo, 0.0, 0.0, 0.0, 0)));
        }
        assert!(ol.is_full());
        assert!(!ol.push_top(Order::set(OrderType::MoveTo, 0.0, 0.0, 0.0, 0)));
    }

    #[test]
    fn push_top_is_most_recent_first() {
        let mut ol = OrderList::new();
        ol.push_top(Order::set(OrderType::MoveTo, 1.0, 0.0, 0.0, 0));
        ol.push_top(Order::set(OrderType::Fire, 2.0, 0.0, 0.0, 0));
        assert_eq!(ol.top().unwrap().ty, OrderType::Fire);
    }

    #[test]
    fn remove_type_drops_all_matches() {
        let mut ol = OrderList::new();
        ol.push_top(Order::set(OrderType::MoveTo, 1.0, 0.0, 0.0, 0));
        ol.push_top(Order::set(OrderType::Fire, 0.0, 0.0, 0.0, 0));
        ol.push_top(Order::set(OrderType::MoveTo, 2.0, 0.0, 0.0, 0));
        ol.remove_type(OrderType::MoveTo);
        assert_eq!(ol.len(), 1);
        assert_eq!(ol.top().unwrap().ty, OrderType::Fire);
    }
}
