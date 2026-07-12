//! Port of `Logic/MatrixRoadNetwork.{cpp,hpp}` — `CMatrixRoadNetwork`,
//! the zone/road/crotch/place/region graph the robot AI navigates, plus
//! the zone-level pathfinder from `MatrixLogic.cpp` (`SetZoneAccess` /
//! `FindPathInZone`, MatrixLogic.cpp:1437-1800).
//!
//! Runtime scope only: the network is deserialized from the CMAP
//! `roads/Data` blob (version 27) and queried; the map-editor build
//! functions (UnionRoad, SplitRoad, CreateAdditionalRoad, ...) are not
//! ported.
//!
//! C++ pointers become indices: roads/crotches refer to each other via
//! `usize` indices into `RoadNetwork::{roads, crotches}`; the intrusive
//! linked lists become `Vec`s in the same first→last order, so linear
//! searches visit elements in identical order.
//!
//! Wire format (CBuf, little-endian): `Bool()` is 1 byte (MSVC
//! `sizeof(bool)`), `Byte()` 1, `Int()` 4 (i32), `Dword()` 4 (u32).
//! `AnyStruct<CPoint>` is the raw 8-byte struct (2×i32) and
//! `AnyStruct<CRect>` 16 bytes (4×i32) — `CPoint`/`CRect` wrap
//! `tagPOINT`/`tagRECT` (BaseDef.hpp:24,40) and contain only LONGs, so
//! /Zp1 packing doesn't change their layout.

use anyhow::{Result, bail};

/// `Base::CPoint` (BaseDef.hpp:24) — 2×i32 on the wire.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Point {
    pub x: i32,
    pub y: i32,
}

impl Point {
    pub fn new(x: i32, y: i32) -> Self {
        Self { x, y }
    }

    /// `CPoint::Dist2` (BaseDef.hpp:36).
    pub fn dist2(&self, p: &Point) -> i32 {
        (p.x - self.x) * (p.x - self.x) + (p.y - self.y) * (p.y - self.y)
    }
}

/// `Base::CRect` (BaseDef.hpp:40) — 4×i32 on the wire.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Rect {
    pub left: i32,
    pub top: i32,
    pub right: i32,
    pub bottom: i32,
}

/// `SMatrixMapZone` (MatrixRoadNetwork.hpp:94). The editor-only color
/// fields are dropped; `near_zone*` are parallel arrays exactly like
/// the C++ `m_NearZone` / `m_NearZoneConnectSize` / `m_NearZoneMove`
/// (cnt ≤ 64), `place` holds up to 4 entries (`m_Place[4]`).
#[derive(Clone, Debug, Default)]
pub struct Zone {
    pub center: Point,
    pub size: i32,
    pub perim: i32,
    pub rect: Rect,
    /// `m_Move` — bit `1<<chassis` set = impassable for that chassis.
    pub move_mask: u8,
    pub road: bool,
    pub bottleneck: bool,
    pub access: bool,
    pub near_zone: Vec<i32>,
    pub near_zone_connect_size: Vec<i32>,
    pub near_zone_move: Vec<u8>,
    pub place: Vec<i32>,
    pub region: i32,
    // m_FPLevel / m_FPWt / m_FPWtp — wave-search scratch.
    pub fp_level: i32,
    pub fp_wt: i32,
    pub fp_wtp: i32,
}

/// `CMatrixCrotch` (MatrixRoadNetwork.hpp:65) — a road junction.
#[derive(Clone, Debug)]
pub struct Crotch {
    pub center: Point,
    pub zone: i32,
    /// Indices into `RoadNetwork::roads`; `m_Road[16]`, cnt ≤ 16.
    pub roads: Vec<usize>,
    pub island: i32,
    pub region: i32,
    /// `m_Data` — BFS scratch.
    pub data: i32,
    pub select: bool,
    pub critical: bool,
}

impl Default for Crotch {
    /// Mirrors the C++ constructor (MatrixRoadNetwork.cpp:319).
    fn default() -> Self {
        Self {
            center: Point::default(),
            zone: -1,
            roads: Vec::new(),
            island: 0,
            region: -1,
            data: 0,
            select: false,
            critical: false,
        }
    }
}

/// `CMatrixRoad` (MatrixRoadNetwork.hpp:13). `m_Path` is editor-only
/// (never serialized by Save/Load) and is omitted.
#[derive(Clone, Debug)]
pub struct Road {
    /// Crotch indices.
    pub start: usize,
    pub end: usize,
    /// Zone indices along the road, start→end.
    pub zones: Vec<i32>,
    pub dist: i32,
    pub move_mask: u8,
    /// `m_Data` — calc scratch.
    pub data: i32,
}

impl Road {
    /// `GetOtherCrotch` (MatrixRoadNetwork.hpp:62).
    pub fn other_crotch(&self, crotch: usize) -> usize {
        if self.start != crotch { self.start } else { self.end }
    }
}

/// `SMatrixRoadRouteHeader` (MatrixRoadNetwork.hpp:133).
#[derive(Clone, Copy, Debug, Default)]
pub struct RouteHeader {
    pub cnt: i32,
    pub dist: i32,
}

/// `SMatrixRoadRouteUnit` (MatrixRoadNetwork.hpp:128).
#[derive(Clone, Copy, Debug, Default)]
pub struct RouteUnit {
    pub road: Option<usize>,
    pub crotch: usize,
}

/// `CMatrixRoadRoute` (MatrixRoadNetwork.hpp:138) — a set of crotch
/// paths. `units` is a flat `list_cnt_max × crotch_cnt` matrix, stride
/// = the network's crotch count at allocation time (passed to every
/// method as `crotch_cnt`, matching the C++ which always indexes via
/// the live `m_Parent->m_CrotchCnt`).
#[derive(Clone, Debug, Default)]
pub struct RoadRoute {
    pub start: Option<usize>,
    pub end: Option<usize>,
    pub list_cnt: usize,
    list_cnt_max: usize,
    pub header: Vec<RouteHeader>,
    pub units: Vec<RouteUnit>,
}

impl RoadRoute {
    /// `Clear` (MatrixRoadNetwork.cpp:410) — frees storage.
    pub fn clear(&mut self) {
        self.start = None;
        self.end = None;
        self.list_cnt = 0;
        self.list_cnt_max = 0;
        self.header = Vec::new();
        self.units = Vec::new();
    }

    /// `ClearFast` (MatrixRoadNetwork.cpp:420) — keeps storage.
    pub fn clear_fast(&mut self) {
        self.start = None;
        self.end = None;
        self.list_cnt = 0;
    }

    /// `NeedEmpty` (MatrixRoadNetwork.cpp:427). Growth formula kept
    /// verbatim, including the odd `strong` adjustments.
    pub fn need_empty(&mut self, crotch_cnt: usize, cnt: usize, strong: bool) {
        if self.list_cnt + cnt <= self.list_cnt_max {
            return;
        }
        self.list_cnt_max += self.list_cnt + cnt;
        if strong {
            if self.list_cnt_max <= 1 {
            } else if self.list_cnt_max <= 4 {
                self.list_cnt_max = 4;
            } else {
                self.list_cnt_max += 8;
            }
        }
        // HAllocClearEx semantics: existing entries preserved, new ones
        // zeroed.
        self.header.resize(self.list_cnt_max, RouteHeader::default());
        self.units
            .resize(self.list_cnt_max * crotch_cnt, RouteUnit::default());
    }

    /// `AddList` (MatrixRoadNetwork.cpp:441). Only `cnt` is reset —
    /// `dist` keeps whatever was in the (zero-initialized or reused)
    /// slot, exactly like the C++.
    pub fn add_list(&mut self, crotch_cnt: usize) -> usize {
        self.need_empty(crotch_cnt, 1, false);
        self.header[self.list_cnt].cnt = 0;
        self.list_cnt += 1;
        self.list_cnt - 1
    }

    /// `CopyUnit` (MatrixRoadNetwork.cpp:449).
    pub fn copy_unit(&mut self, crotch_cnt: usize, listfrom: usize, listto: usize) {
        debug_assert!(listfrom < self.list_cnt);
        debug_assert!(listto < self.list_cnt);
        self.header[listto] = self.header[listfrom];
        let n = self.header[listto].cnt as usize;
        let (src, dst) = (listfrom * crotch_cnt, listto * crotch_cnt);
        for i in 0..n {
            self.units[dst + i] = self.units[src + i];
        }
    }

    /// `AddUnit` (MatrixRoadNetwork.cpp:458).
    pub fn add_unit(
        &mut self,
        crotch_cnt: usize,
        listno: usize,
        road: Option<usize>,
        crotch: usize,
    ) -> usize {
        debug_assert!(listno < self.list_cnt);
        debug_assert!((self.header[listno].cnt as usize) < crotch_cnt);
        let at = listno * crotch_cnt + self.header[listno].cnt as usize;
        self.units[at] = RouteUnit {
            road,
            crotch,
        };
        self.header[listno].cnt += 1;
        self.header[listno].cnt as usize - 1
    }
}

/// `SMatrixPlace` (MatrixRoadNetwork.hpp:164). `near` / `near_move`
/// are the parallel `m_Near[16]` / `m_NearMove[16]` arrays.
#[derive(Clone, Debug, Default)]
pub struct Place {
    pub pos: Point,
    pub move_mask: u8,
    pub border_left: u8,
    pub border_top: u8,
    pub border_right: u8,
    pub border_bottom: u8,
    pub border_cnt: u8,
    pub edge_left: u8,
    pub edge_top: u8,
    pub edge_right: u8,
    pub edge_bottom: u8,
    pub edge_cnt: u8,
    pub blockade: [bool; 9],
    pub cannon: u8,
    pub region: i32,
    pub near: Vec<i32>,
    pub near_move: Vec<u8>,
    // Editor-side fields, kept because the place free-list uses them.
    pub empty: bool,
    pub empty_next: i32,
    pub rank: i32,
    // Game-side scratch. `m_Data` is an int in C++ but is only ever
    // combined with dword masks (ActionDataPL), so u32 here.
    pub data: u32,
    pub underfire: u8,
}

/// `SMatrixPlaceList` (MatrixRoadNetwork.hpp:189) — one 64×64-cell
/// bucket of the place lookup grid.
#[derive(Clone, Copy, Debug, Default)]
pub struct PlaceList {
    pub sme: i32,
    pub cnt: i32,
}

/// `SMatrixRegion` (MatrixRoadNetwork.hpp:194). `crotch` holds crotch
/// indices (`m_Crotch[16]`), `place`/`place_all` place indices
/// (`m_Place[16]` / `m_PlaceAll[64]`).
#[derive(Clone, Debug, Default)]
pub struct Region {
    pub center: Point,
    pub radius_place: i32,
    pub crotch: Vec<usize>,
    pub place: Vec<i32>,
    pub place_all: Vec<i32>,
    pub near: Vec<i32>,
    pub near_move: Vec<u8>,
    pub fp_level: i32,
    pub fp_wt: i32,
    pub fp_wtp: i32,
    pub color: u32,
}

/// Sequential little-endian reader over the `roads/Data` blob — stands
/// in for the CBuf read cursor (`TestGet` → error instead of ERROR_E).
struct Reader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        if self.pos + n > self.data.len() {
            bail!(
                "road network blob overrun: need {} bytes at offset {} of {}",
                n,
                self.pos,
                self.data.len()
            );
        }
        let s = &self.data[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }

    fn byte(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }

    fn bool_(&mut self) -> Result<bool> {
        Ok(self.take(1)?[0] != 0)
    }

    fn int(&mut self) -> Result<i32> {
        let s = self.take(4)?;
        Ok(i32::from_le_bytes([s[0], s[1], s[2], s[3]]))
    }

    fn dword(&mut self) -> Result<u32> {
        let s = self.take(4)?;
        Ok(u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
    }

    fn point(&mut self) -> Result<Point> {
        Ok(Point {
            x: self.int()?,
            y: self.int()?,
        })
    }

    fn rect(&mut self) -> Result<Rect> {
        Ok(Rect {
            left: self.int()?,
            top: self.int()?,
            right: self.int()?,
            bottom: self.int()?,
        })
    }
}

/// `CMatrixRoadNetwork` (MatrixRoadNetwork.hpp:217).
#[derive(Debug, Default)]
pub struct RoadNetwork {
    pub zones: Vec<Zone>,
    pub crotches: Vec<Crotch>,
    pub roads: Vec<Road>,
    /// Route cache storage. NOTE: the C++ `AddRoute`
    /// (MatrixRoadNetwork.cpp:1775) calls LIST_DEL on the fresh node —
    /// a no-op for an unlinked node — so routes are never linked into
    /// `m_RouteFirst..m_RouteLast` and `FindRoute` never sees them. We
    /// keep routes here so they're owned rather than leaked, but
    /// `find_route` still never matches because `calc_route` leaves
    /// `start`/`end` unset, exactly like the C++.
    pub routes: Vec<RoadRoute>,
    pub places: Vec<Place>,
    /// Head of the place free list (`m_PlaceEmpty`), -1 = none.
    pub place_empty: i32,
    pub regions: Vec<Region>,
    /// Place lookup grid, `pl_size_x × pl_size_y` row-major.
    pub pl_list: Vec<PlaceList>,
    pub pl_size_x: i32,
    pub pl_size_y: i32,
    pub pl_shift: i32,
    pub pl_mask: i32,
    // Scratch buffers (m_RoadFindIndex / m_CrotchFindIndex /
    // m_RegionFindIndex on CMatrixRoadNetwork; m_ZoneIndex /
    // m_ZoneIndexAccess on CMatrixMapLogic).
    road_find_index: Vec<usize>,
    crotch_find_index: Vec<usize>,
    region_find_index: Vec<i32>,
    zone_index: Vec<i32>,
    zone_index_access: Vec<i32>,
}

impl RoadNetwork {
    /// Mirrors the C++ constructor (MatrixRoadNetwork.cpp:473):
    /// `m_PLShift = 6` (64-cell buckets), `m_PlaceEmpty = -1`.
    pub fn new() -> Self {
        Self {
            place_empty: -1,
            pl_shift: 6,
            pl_mask: (1 << 6) - 1,
            ..Default::default()
        }
    }

    /// `Clear` (MatrixRoadNetwork.cpp:518).
    pub fn clear(&mut self) {
        self.road_find_index = Vec::new();
        self.crotch_find_index = Vec::new();
        self.region_find_index = Vec::new();
        self.zone_index = Vec::new();
        self.zone_index_access = Vec::new();
        self.clear_pl();
        self.clear_region();
        self.clear_place();
        self.clear_route();
        self.zones = Vec::new();
        self.roads = Vec::new();
        self.crotches = Vec::new();
    }

    // ── Load ────────────────────────────────────────────────────────

    /// Port of `Load` (MatrixRoadNetwork.cpp:2561) for ver==27 — the
    /// game refuses any other version (MatrixMapPrepare.cpp:1606), so
    /// every `ver >= 21/24/25/26/27` branch is collapsed to
    /// always-taken. `data` is the blob *after* the leading version
    /// dword.
    pub fn load(&mut self, data: &[u8]) -> Result<()> {
        self.clear();
        let mut b = Reader::new(data);

        let cnt = b.int()?;
        if cnt < 0 {
            bail!("road network: negative zone count {}", cnt);
        }
        self.zones = vec![Zone::default(); cnt as usize];
        for i in 0..cnt as usize {
            let z = &mut self.zones[i];
            z.center = b.point()?;
            z.bottleneck = b.bool_()?;
            z.size = b.int()?;
            z.perim = b.int()?;
            z.rect = b.rect()?;
            z.road = b.bool_()?;
            z.move_mask = b.byte()?;
            let near_cnt = b.byte()? as usize;
            for _ in 0..near_cnt {
                z.near_zone.push(b.int()?);
                z.near_zone_connect_size.push(b.int()?);
                z.near_zone_move.push(b.byte()?);
            }
            z.region = b.int()?;
            let place_cnt = b.byte()? as usize;
            for _ in 0..place_cnt {
                z.place.push(b.int()?);
            }
            z.access = false;
        }

        let cnt = b.int()?;
        for _ in 0..cnt {
            let center = b.point()?;
            let zone = b.int()?;
            let critical = b.bool_()?;
            let island = b.int()?;
            // C++: crotch->m_Region = m_Zone[crotch->m_Zone].m_Region.
            let region = self.zones[zone as usize].region;
            self.crotches.push(Crotch {
                center,
                zone,
                critical,
                island,
                region,
                ..Default::default()
            });
        }

        let cnt = b.int()?;
        for _ in 0..cnt {
            let start = b.int()? as usize;
            let end = b.int()? as usize;
            let zone_cnt = b.int()? as usize;
            let mut zones = Vec::with_capacity(zone_cnt);
            for _ in 0..zone_cnt {
                zones.push(b.int()?);
            }
            let dist = b.int()?;
            let move_mask = b.byte()?;

            let ri = self.roads.len();
            self.roads.push(Road {
                start,
                end,
                zones,
                dist,
                move_mask,
                data: 0,
            });
            // AddRoad asserts m_RoadCnt < 16 (MatrixRoadNetwork.cpp:370).
            debug_assert!(self.crotches[start].roads.len() < 16);
            self.crotches[start].roads.push(ri);
            debug_assert!(self.crotches[end].roads.len() < 16);
            self.crotches[end].roads.push(ri);
        }

        let cnt = b.int()?;
        self.set_place_cnt(cnt);
        self.place_empty = -1;
        for i in 0..cnt as usize {
            let p = &mut self.places[i];
            p.empty = false;
            p.empty_next = -1;

            p.pos = b.point()?;
            p.move_mask = b.byte()?;
            p.border_left = b.byte()?;
            p.border_top = b.byte()?;
            p.border_right = b.byte()?;
            p.border_bottom = b.byte()?;
            p.border_cnt = b.byte()?;
            p.edge_left = b.byte()?;
            p.edge_top = b.byte()?;
            p.edge_right = b.byte()?;
            p.edge_bottom = b.byte()?;
            p.edge_cnt = b.byte()?;
            for u in 0..9 {
                p.blockade[u] = b.bool_()?;
            }
            p.cannon = b.byte()?;
            if p.cannon != 0 {
                p.move_mask = 1 + 2 + 4 + 8 + 16;
            }
            let near_cnt = b.byte()? as usize;
            for _ in 0..near_cnt {
                p.near.push(b.int()?);
                p.near_move.push(b.byte()?);
            }
            p.region = -1;
        }

        let cnt = b.int()?;
        self.regions = vec![Region::default(); cnt.max(0) as usize];
        for i in 0..cnt as usize {
            let center = b.point()?;
            let radius_place = b.int()?;

            let crotch_cnt = b.byte()? as usize;
            let mut crotch = Vec::with_capacity(crotch_cnt);
            for _ in 0..crotch_cnt {
                crotch.push(b.int()? as usize);
            }

            let place_cnt = b.byte()? as usize;
            let mut place = Vec::with_capacity(place_cnt);
            for _ in 0..place_cnt {
                let pl = b.int()?;
                place.push(pl);
                self.places[pl as usize].region = i as i32;
            }

            let place_all_cnt = b.byte()? as usize;
            let mut place_all = Vec::with_capacity(place_all_cnt);
            for _ in 0..place_all_cnt {
                place_all.push(b.int()?);
            }

            let near_cnt = b.byte()? as usize;
            let mut near = Vec::with_capacity(near_cnt);
            let mut near_move = Vec::with_capacity(near_cnt);
            for _ in 0..near_cnt {
                near.push(b.int()?);
                near_move.push(b.byte()?);
            }

            let color = b.dword()?;

            self.regions[i] = Region {
                center,
                radius_place,
                crotch,
                place,
                place_all,
                near,
                near_move,
                fp_level: 0,
                fp_wt: 0,
                fp_wtp: 0,
                color,
            };
        }

        Ok(())
    }

    // ── Place lookup grid (PL) ──────────────────────────────────────

    /// `ClearPL` (MatrixRoadNetwork.cpp:2340).
    pub fn clear_pl(&mut self) {
        self.pl_list = Vec::new();
        self.pl_size_x = 0;
        self.pl_size_y = 0;
    }

    /// `InitPL` (MatrixRoadNetwork.cpp:2347) — buckets the places into
    /// a 64×64-cell grid, reordering `places` so each bucket is a
    /// contiguous run, then remaps every stored place index (regions,
    /// zones, place near-lists) through `changei`.
    pub fn init_pl(&mut self, mapsizex: i32, mapsizey: i32) {
        self.clear_pl();

        self.pl_size_x =
            (mapsizex >> self.pl_shift) + if mapsizex & self.pl_mask != 0 { 1 } else { 0 };
        self.pl_size_y =
            (mapsizey >> self.pl_shift) + if mapsizey & self.pl_mask != 0 { 1 } else { 0 };

        // C++ checks m_PLSizeX twice (the second was clearly meant to
        // be Y); behavior is identical so we just check X once.
        if self.places.is_empty() || self.pl_size_x <= 0 {
            return;
        }

        let place_cnt = self.places.len();
        let mut nap: Vec<Place> = Vec::with_capacity(place_cnt);
        let mut changei = vec![0i32; place_cnt];
        self.pl_list =
            vec![PlaceList::default(); (self.pl_size_x * self.pl_size_y) as usize];

        let mut napcnt: i32 = 0;
        for y in 0..self.pl_size_y {
            for x in 0..self.pl_size_x {
                let pl = &mut self.pl_list[(y * self.pl_size_x + x) as usize];
                pl.sme = napcnt;
                for (i, place) in self.places.iter().enumerate() {
                    if (place.pos.x >> self.pl_shift) == x && (place.pos.y >> self.pl_shift) == y {
                        let mut p = place.clone();
                        p.empty = false;
                        p.empty_next = -1;
                        changei[i] = napcnt;
                        nap.push(p);
                        napcnt += 1;
                        pl.cnt += 1;
                    }
                }
            }
        }
        assert_eq!(napcnt as usize, place_cnt);

        for region in self.regions.iter_mut() {
            for p in region.place.iter_mut() {
                *p = changei[*p as usize];
            }
            for p in region.place_all.iter_mut() {
                *p = changei[*p as usize];
            }
        }

        self.places = nap;
        self.place_empty = -1;

        for zone in self.zones.iter_mut() {
            for p in zone.place.iter_mut() {
                *p = changei[*p as usize];
            }
        }
        for place in self.places.iter_mut() {
            for n in place.near.iter_mut() {
                *n = changei[*n as usize];
            }
        }
    }

    /// `CorrectRectPL` (MatrixRoadNetwork.cpp:2410).
    pub fn correct_rect_pl(&self, mapcoords: &Rect) -> Rect {
        Rect {
            left: 0.max(mapcoords.left >> self.pl_shift),
            top: 0.max(mapcoords.top >> self.pl_shift),
            right: self.pl_size_x.min((mapcoords.right >> self.pl_shift) + 1),
            bottom: self.pl_size_y.min((mapcoords.bottom >> self.pl_shift) + 1),
        }
    }

    /// `ActionDataPL` (MatrixRoadNetwork.cpp:2420) — applies
    /// `data = (data & mask_and) | mask_or` to every place whose grid
    /// bucket intersects `mapcoords`.
    pub fn action_data_pl(&mut self, mapcoords: &Rect, mask_and: u32, mask_or: u32) {
        let sx = 0.max(mapcoords.left >> self.pl_shift);
        let sy = 0.max(mapcoords.top >> self.pl_shift);
        let ex = self.pl_size_x.min((mapcoords.right >> self.pl_shift) + 1);
        let ey = self.pl_size_y.min((mapcoords.bottom >> self.pl_shift) + 1);

        for y in sy..ey {
            for x in sx..ex {
                let pl = self.pl_list[(y * self.pl_size_x + x) as usize];
                for i in 0..pl.cnt {
                    let mp = &mut self.places[(pl.sme + i) as usize];
                    mp.data = (mp.data & mask_and) | mask_or;
                }
            }
        }
    }

    /// `FindInPL` (MatrixRoadNetwork.cpp:2436) — place index whose 4×4
    /// footprint contains `mappos`, or -1.
    pub fn find_in_pl(&self, mappos: &Point) -> i32 {
        let plr = self.correct_rect_pl(&Rect {
            left: mappos.x - 4,
            top: mappos.y - 4,
            right: mappos.x + 4,
            bottom: mappos.y + 4,
        });
        for y in plr.top..plr.bottom {
            for x in plr.left..plr.right {
                let plist = self.pl_list[(y * self.pl_size_x + x) as usize];
                for u in 0..plist.cnt {
                    let place = &self.places[(plist.sme + u) as usize];
                    if mappos.x >= place.pos.x
                        && mappos.y >= place.pos.y
                        && mappos.x < place.pos.x + 4
                        && mappos.y < place.pos.y + 4
                    {
                        return plist.sme + u;
                    }
                }
            }
        }
        -1
    }

    // ── Places ──────────────────────────────────────────────────────

    /// `ClearPlace` (MatrixRoadNetwork.cpp:1992).
    pub fn clear_place(&mut self) {
        self.places = Vec::new();
        self.place_empty = -1;
    }

    /// `GetPlace` (MatrixRoadNetwork.hpp:318).
    pub fn get_place(&self, no: i32) -> &Place {
        &self.places[no as usize]
    }

    pub fn get_place_mut(&mut self, no: i32) -> &mut Place {
        &mut self.places[no as usize]
    }

    /// `SetPlaceCnt` (MatrixRoadNetwork.cpp:1999) — grow/shrink the
    /// place array, maintaining the free list. Newly grown entries are
    /// pushed onto the free list head.
    pub fn set_place_cnt(&mut self, cnt: i32) {
        let old = self.places.len() as i32;
        if old == cnt {
            return;
        }
        self.places.resize(cnt.max(0) as usize, Place::default());
        if cnt < old {
            if self.place_empty >= cnt {
                self.place_empty = -1;
            } else if self.place_empty >= 0 {
                // The C++ walk (cpp:2007-2011) follows the just-cleared
                // -1 link into m_Place[-1]; the i>=0 guard keeps the
                // same chain result without the out-of-bounds read.
                let mut i = self.place_empty;
                while i >= 0 && self.places[i as usize].empty_next >= 0 {
                    if self.places[i as usize].empty_next >= cnt {
                        self.places[i as usize].empty_next = -1;
                    }
                    i = self.places[i as usize].empty_next;
                }
            }
        } else {
            for i in old..cnt {
                self.places[i as usize].empty = true;
                self.places[i as usize].empty_next = self.place_empty;
                self.place_empty = i;
            }
        }
    }

    /// `AllocPlace` (MatrixRoadNetwork.cpp:2023).
    pub fn alloc_place(&mut self) -> i32 {
        if self.place_empty < 0 {
            let cnt = self.places.len() as i32 + 256;
            self.set_place_cnt(cnt);
        }
        let no = self.place_empty;
        self.place_empty = self.places[no as usize].empty_next;

        // ZeroMemory in C++.
        self.places[no as usize] = Place::default();
        self.places[no as usize].empty = false;
        self.places[no as usize].empty_next = -1;
        no
    }

    /// `FreePlace` (MatrixRoadNetwork.cpp:2035).
    pub fn free_place(&mut self, no: i32) {
        debug_assert!(no >= 0 && (no as usize) < self.places.len());
        debug_assert!(!self.places[no as usize].empty);
        self.places[no as usize].empty = true;
        self.places[no as usize].empty_next = self.place_empty;
        self.place_empty = no;
    }

    // ── Zones / crotches / roads lookups ────────────────────────────

    /// `FindNearZoneByCenter` (MatrixRoadNetwork.cpp:855) — returns
    /// `(zone index or -1, dist2)`.
    pub fn find_near_zone_by_center(&self, p: &Point) -> (i32, i32) {
        let mut rv = -1;
        let mut mz = 0;
        if !self.zones.is_empty() {
            rv = 0;
            mz = p.dist2(&self.zones[0].center);
            for (i, z) in self.zones.iter().enumerate().skip(1) {
                let cz = p.dist2(&z.center);
                if cz < mz {
                    rv = i as i32;
                    mz = cz;
                }
            }
        }
        (rv, mz)
    }

    /// `FindCrotchByZone` (MatrixRoadNetwork.cpp:618).
    pub fn find_crotch_by_zone(&self, zone: i32) -> Option<usize> {
        self.crotches.iter().position(|c| c.zone == zone)
    }

    /// `FindRoadByZone` (MatrixRoadNetwork.cpp:572).
    pub fn find_road_by_zone(&self, zone: i32) -> Option<usize> {
        self.roads
            .iter()
            .position(|r| r.zones.iter().any(|&z| z == zone))
    }

    /// `Unselect` (MatrixRoadNetwork.cpp:1248).
    pub fn unselect(&mut self) {
        for c in self.crotches.iter_mut() {
            c.select = false;
        }
    }

    // ── Zone-graph wave search along roads ──────────────────────────

    /// `CalcPathZoneByRoad` (MatrixRoadNetwork.cpp:1257) — wave search
    /// over road zones from `zend` backwards, then a greedy descend
    /// from `zstart` writing the zone path. Returns the path length
    /// (0 = same zone or unreachable). `path` must hold `zones.len()`
    /// entries.
    pub fn calc_path_zone_by_road(&mut self, zstart: i32, zend: i32, path: &mut [i32]) -> usize {
        if zstart == zend {
            return 0;
        }
        debug_assert!(self.zones[zstart as usize].road);
        debug_assert!(self.zones[zend as usize].road);

        for z in self.zones.iter_mut() {
            z.fp_level = -1;
            z.fp_wt = 1;
            z.fp_wtp = 0;
        }

        // The C++ uses two fixed ZoneCnt-sized arrays; a zone can be
        // re-queued when its weight improves, so Vec+push avoids the
        // (theoretical) C++ overflow without changing the visit order.
        let mut fp1: Vec<i32> = Vec::with_capacity(self.zones.len());
        let mut fp2: Vec<i32> = Vec::with_capacity(self.zones.len());
        let mut level = 0;

        fp1.push(zend);
        self.zones[zend as usize].fp_level = level;
        self.zones[zend as usize].fp_wtp = 0;
        level += 1;

        let mut fok = false;

        while !fp1.is_empty() {
            for i in 0..fp1.len() {
                let zi = fp1[i] as usize;
                for u in 0..self.zones[zi].near_zone.len() {
                    let nz = self.zones[zi].near_zone[u];
                    let nzu = nz as usize;
                    if !self.zones[nzu].road {
                        continue;
                    }
                    let wtp = self.zones[zi].fp_wtp + self.zones[nzu].fp_wt;
                    if self.zones[nzu].fp_level < 0 || wtp < self.zones[nzu].fp_wtp {
                        self.zones[nzu].fp_level = level;
                        self.zones[nzu].fp_wtp = wtp;
                        fp2.push(nz);
                        if zstart == nz {
                            fok = true;
                        }
                    }
                }
            }
            level += 1;
            std::mem::swap(&mut fp1, &mut fp2);
            fp2.clear();
        }

        if !fok {
            return 0;
        }

        path[0] = zstart;
        let mut cnt = 1usize;
        let mut curzone = zstart;
        let mut curwt = self.zones[curzone as usize].fp_wtp;
        loop {
            let zi = curzone as usize;
            let mut nz_: i32 = -1;
            let mut wtp_ = curwt;
            for i in 0..self.zones[zi].near_zone.len() {
                let nz = self.zones[zi].near_zone[i];
                if !self.zones[nz as usize].road {
                    continue;
                }
                let wtp = self.zones[nz as usize].fp_wtp;
                if nz == zend {
                    nz_ = nz;
                    wtp_ = wtp;
                    break;
                } else if wtp <= wtp_ {
                    nz_ = nz;
                    wtp_ = wtp;
                }
            }
            if nz_ < 0 {
                panic!("calc_path_zone_by_road: descend dead end (C++ ERROR_E)");
            }
            curzone = nz_;
            curwt = wtp_;
            debug_assert!(cnt + 1 <= self.zones.len());
            path[cnt] = curzone;
            cnt += 1;
            if curzone == zend {
                break;
            }
        }
        cnt
    }

    /// `CalcDistByRoad` (MatrixRoadNetwork.cpp:1352). Returns -1 when
    /// no path exists (including zstart==zend, like the C++).
    pub fn calc_dist_by_road(
        &mut self,
        zstart: i32,
        zend: i32,
        path: Option<&mut [i32]>,
    ) -> i32 {
        debug_assert!(!self.zones.is_empty());
        let mut own_buf;
        let path: &mut [i32] = match path {
            Some(p) => p,
            None => {
                own_buf = vec![0i32; self.zones.len()];
                &mut own_buf
            }
        };
        let cnt = self.calc_path_zone_by_road(zstart, zend, path);
        if cnt == 0 {
            return -1;
        }
        let mut rv = 0;
        for i in 1..cnt {
            let d2 = self.zones[path[i - 1] as usize]
                .center
                .dist2(&self.zones[path[i] as usize].center);
            rv += (d2 as f64).sqrt() as i32;
        }
        rv
    }

    /// `CalcDistRoad` (MatrixRoadNetwork.cpp:1376) +
    /// `CMatrixRoad::CalcDist` (cpp:308) — refresh every road's dist
    /// from its zone-center chain.
    pub fn calc_dist_road(&mut self) {
        for ri in 0..self.roads.len() {
            let mut dist = 0;
            for i in 1..self.roads[ri].zones.len() {
                let a = self.zones[self.roads[ri].zones[i - 1] as usize].center;
                let b = self.zones[self.roads[ri].zones[i] as usize].center;
                dist += (a.dist2(&b) as f64).sqrt() as i32;
            }
            self.roads[ri].dist = dist;
        }
    }

    // ── Route cache ─────────────────────────────────────────────────

    /// `ClearRoute` (MatrixRoadNetwork.cpp:1764).
    pub fn clear_route(&mut self) {
        self.routes = Vec::new();
    }

    /// `DeleteRoute` (MatrixRoadNetwork.cpp:1769). Indices of routes
    /// after `idx` shift down (the C++ used stable pointers; at
    /// runtime only ClearRoute deletes, so this never matters).
    pub fn delete_route(&mut self, idx: usize) {
        self.routes.remove(idx);
    }

    /// `AddRoute` (MatrixRoadNetwork.cpp:1775) — see the `routes`
    /// field doc for the C++ LIST_DEL quirk.
    pub fn add_route(&mut self) -> usize {
        self.routes.push(RoadRoute::default());
        self.routes.len() - 1
    }

    /// `CalcRoute` (MatrixRoadNetwork.cpp:1811) — depth-first
    /// enumeration of all crotch paths from `cstart` to `cend` no
    /// longer than 2× the direct road distance. Returns the route's
    /// index in `routes`.
    pub fn calc_route(&mut self, cstart: usize, cend: usize) -> usize {
        self.unselect();

        let maxdist = self.calc_dist_by_road(
            self.crotches[cstart].zone,
            self.crotches[cend].zone,
            None,
        ) * 2;

        let idx = self.add_route();
        let mut route = std::mem::take(&mut self.routes[idx]);
        route.add_list(self.crotches.len());
        self.calc_route_r(cstart, cend, &mut route, None, cstart, maxdist);
        route.list_cnt -= 1;
        self.routes[idx] = route;
        idx
    }

    /// `CalcRoute_r` (MatrixRoadNetwork.cpp:1782).
    fn calc_route_r(
        &mut self,
        cstart: usize,
        cend: usize,
        route: &mut RoadRoute,
        road: Option<usize>,
        crotch: usize,
        maxdist: i32,
    ) {
        let cc = self.crotches.len();
        route.add_unit(cc, route.list_cnt - 1, road, crotch);
        if let Some(r) = road {
            route.header[route.list_cnt - 1].dist += self.roads[r].dist;
        }
        if crotch == cend {
            if route.header[route.list_cnt - 1].cnt > 1 {
                route.add_list(cc);
                route.copy_unit(cc, route.list_cnt - 2, route.list_cnt - 1);
            }
            if let Some(r) = road {
                route.header[route.list_cnt - 1].dist -= self.roads[r].dist;
            }
            route.header[route.list_cnt - 1].cnt -= 1;
            return;
        }
        self.crotches[crotch].select = true;

        for i in 0..self.crotches[crotch].roads.len() {
            let ri = self.crotches[crotch].roads[i];
            let cin = self.roads[ri].other_crotch(crotch);
            if self.crotches[cin].select {
                continue;
            }
            if route.header[route.list_cnt - 1].dist + self.roads[ri].dist > maxdist {
                continue;
            }
            if cin == cend || self.crotches[cin].roads.len() > 1 {
                self.calc_route_r(cstart, cend, route, Some(ri), cin, maxdist);
            }
        }

        self.crotches[crotch].select = false;
        if let Some(r) = road {
            route.header[route.list_cnt - 1].dist -= self.roads[r].dist;
        }
        route.header[route.list_cnt - 1].cnt -= 1;
    }

    /// `FindRoute` (MatrixRoadNetwork.cpp:1825).
    pub fn find_route(&self, cstart: usize, cend: usize) -> Option<usize> {
        self.routes
            .iter()
            .position(|r| r.start == Some(cstart) && r.end == Some(cend))
    }

    /// `GetRoute` (MatrixRoadNetwork.cpp:1837) — cache lookup then
    /// recompute.
    pub fn get_route(&mut self, cstart: usize, cend: usize) -> usize {
        match self.find_route(cstart, cend) {
            Some(r) => r,
            None => self.calc_route(cstart, cend),
        }
    }

    // ── Crotch-graph path to a region ───────────────────────────────

    /// `FindPathFromCrotchToRegion` (MatrixRoadNetwork.cpp:1850) —
    /// BFS over crotches in three relaxation passes (`type` 0..=2:
    /// stay near the start/target regions + chassis filter, then
    /// chassis filter only, then unrestricted), appending the
    /// road/crotch chain to the last list of `rr`. On total failure
    /// `rr` is cleared. `mm` is the chassis bit (`1<<nsh`); a road
    /// with that bit set is impassable. The debug-only `test`
    /// visualization parameter is dropped.
    pub fn find_path_from_crotch_to_region(
        &mut self,
        mm: u8,
        cstart: usize,
        region: i32,
        rr: &mut RoadRoute,
    ) {
        // cstart itself is never appended to rr.
        if self.crotches[cstart].region == region {
            return;
        }

        if self.road_find_index.len() < self.roads.len() {
            self.road_find_index.resize(self.roads.len(), 0);
        }
        if self.crotch_find_index.len() < self.crotches.len() {
            self.crotch_find_index.resize(self.crotches.len(), 0);
        }

        let stride = self.crotches.len();
        let mut crotchcur: Option<usize> = None;

        for type_ in 0..=2 {
            for c in self.crotches.iter_mut() {
                c.data = 0;
            }

            let mut cnt = 0usize;
            let mut sme = 0usize;
            let mut level = 1;
            self.crotch_find_index[cnt] = cstart;
            self.crotches[cstart].data = level;
            cnt += 1;
            level += 1;

            let mut next = cnt;

            while sme < cnt {
                let crotch = self.crotch_find_index[sme];
                for i in 0..self.crotches[crotch].roads.len() {
                    let ri = self.crotches[crotch].roads[i];
                    let crotch2 = self.roads[ri].other_crotch(crotch);
                    if self.crotches[crotch2].data != 0 {
                        continue;
                    }
                    // type 0: don't wander into unrelated regions.
                    if type_ == 0
                        && self.crotches[crotch2].region != self.crotches[cstart].region
                        && self.crotches[crotch2].region != region
                        && self.crotches[crotch2].region >= 0
                    {
                        continue;
                    }
                    // types 0,1: skip roads any of the robots can't pass.
                    if (type_ == 0 || type_ == 1) && self.roads[ri].move_mask & mm != 0 {
                        continue;
                    }

                    debug_assert!(cnt < self.crotches.len());

                    self.crotches[crotch2].data = level;
                    self.crotch_find_index[cnt] = crotch2;
                    cnt += 1;

                    if self.crotches[crotch2].region == region {
                        crotchcur = Some(crotch2);
                        break;
                    }
                }
                if crotchcur.is_some() {
                    break;
                }

                sme += 1;
                if sme >= next {
                    next = cnt;
                    level += 1;
                }
            }

            let Some(found) = crotchcur else {
                continue;
            };

            // Backtrack along decreasing m_Data levels.
            let mut cur = found;
            self.crotch_find_index[0] = cur;
            let mut cnt = 1usize;

            while cur != cstart {
                let mut step: Option<(usize, usize)> = None; // (crotch2, road2)
                for i in 0..self.crotches[cur].roads.len() {
                    let ri = self.crotches[cur].roads[i];
                    let c = self.roads[ri].other_crotch(cur);
                    if self.crotches[c].data != 0 && self.crotches[c].data < level {
                        step = Some((c, ri));
                        break;
                    }
                }
                let Some((crotch2, road2)) = step else {
                    panic!("find_path_from_crotch_to_region: broken backtrack (C++ ERROR_E)");
                };

                debug_assert!(cnt < self.crotches.len());
                debug_assert!(cnt < self.roads.len());

                self.road_find_index[cnt - 1] = road2;
                self.crotch_find_index[cnt] = crotch2;
                cnt += 1;

                cur = crotch2;
                level = self.crotches[cur].data;
            }

            for i in (0..cnt - 1).rev() {
                let listno = rr.list_cnt - 1;
                rr.add_unit(
                    stride,
                    listno,
                    Some(self.road_find_index[i]),
                    self.crotch_find_index[i],
                );
                rr.header[listno].dist += self.roads[self.road_find_index[i]].dist;
            }
            break;
        }
        if crotchcur.is_none() {
            rr.clear();
        }
    }

    /// `FindPathFromRegionPath` (MatrixRoadNetwork.cpp:1961) — chains
    /// `find_path_from_crotch_to_region` over a region list, starting
    /// from the first region's first crotch. An empty `rr.header`
    /// after a hop means the hop failed and cleared the route.
    pub fn find_path_from_region_path(&mut self, mm: u8, rlist: &[i32], rr: &mut RoadRoute) {
        if rlist.len() <= 1 {
            return;
        }

        let cc = self.crotches.len();
        let ln = rr.add_list(cc);
        rr.add_unit(cc, ln, None, self.regions[rlist[0] as usize].crotch[0]);

        for &r in &rlist[1..] {
            let from = rr.units[ln * cc + (rr.header[ln].cnt as usize - 1)].crotch;
            self.find_path_from_crotch_to_region(mm, from, r, rr);
            if rr.header.is_empty() {
                return;
            }
        }
    }

    // ── Regions ─────────────────────────────────────────────────────

    /// `ClearRegion` (MatrixRoadNetwork.cpp:2045).
    pub fn clear_region(&mut self) {
        self.regions = Vec::new();
    }

    /// `GetRegion` (MatrixRoadNetwork.hpp:326).
    pub fn get_region(&self, no: i32) -> &Region {
        &self.regions[no as usize]
    }

    pub fn get_region_mut(&mut self, no: i32) -> &mut Region {
        &mut self.regions[no as usize]
    }

    /// `IsNerestRegion` (MatrixRoadNetwork.cpp:2181).
    pub fn is_nerest_region(&self, r1: i32, r2: i32) -> bool {
        if r1 < 0 || r2 < 0 {
            return false;
        }
        self.regions[r1 as usize].near.iter().any(|&n| n == r2)
    }

    /// `FindNerestRegion` (MatrixRoadNetwork.cpp:2193).
    pub fn find_nerest_region(&self, tp: &Point) -> i32 {
        if self.regions.is_empty() {
            return -1;
        }
        let mut mindist = tp.dist2(&self.regions[0].center);
        let mut r = 0;
        for i in 1..self.regions.len() {
            let t = tp.dist2(&self.regions[i].center);
            if t < mindist {
                mindist = t;
                r = i as i32;
            }
        }
        r
    }

    /// `FindNerestRegionByRadius` (MatrixRoadNetwork.cpp:2214) — first
    /// region whose place radius covers `tp` (checking `curregion`
    /// and its neighbors first), else the region with the closest
    /// center.
    pub fn find_nerest_region_by_radius(&self, tp: &Point, curregion: i32) -> i32 {
        if curregion >= 0 {
            let cur = &self.regions[curregion as usize];
            if cur.radius_place * cur.radius_place >= cur.center.dist2(tp) {
                return curregion;
            }
            for &r in cur.near.iter() {
                let reg = &self.regions[r as usize];
                if reg.radius_place * reg.radius_place >= reg.center.dist2(tp) {
                    return r;
                }
            }
        }

        let mut mindist = 1_000_000_000;
        let mut r = -1;
        for (i, reg) in self.regions.iter().enumerate() {
            let t = reg.center.dist2(tp);
            if reg.radius_place * reg.radius_place >= t {
                return i as i32;
            }
            if t < mindist {
                mindist = t;
                r = i as i32;
            }
        }
        r
    }

    /// `FindPathInRegionInit` (MatrixRoadNetwork.cpp:2244) — reset all
    /// region weights to 1 (the AI raises them for dangerous regions
    /// before running the search).
    pub fn find_path_in_region_init(&mut self) {
        for r in self.regions.iter_mut() {
            r.fp_wt = 1;
        }
    }

    /// `FindPathInRegionRun` (MatrixRoadNetwork.cpp:2250) — weighted
    /// wave from `rend` backwards over the region near-graph (skipping
    /// links whose `near_move` blocks chassis bit `mm`), then a greedy
    /// descend from `rstart`. Returns the region path length, 0 on
    /// failure (or panics if `err`, matching ERROR_E).
    pub fn find_path_in_region_run(
        &mut self,
        mm: u8,
        rstart: i32,
        rend: i32,
        mut path: Option<&mut [i32]>,
        maxpath: i32,
        err: bool,
    ) -> usize {
        if self.regions.len() <= 1 {
            return 0;
        }
        if rstart == rend {
            return 0;
        }

        // C++ sizes the index to RegionCnt; a region can be re-queued
        // when its weight improves, so grow on demand instead of
        // overflowing.
        if self.region_find_index.len() < self.regions.len() {
            self.region_find_index.resize(self.regions.len(), 0);
        }

        for r in self.regions.iter_mut() {
            r.fp_level = -1;
            r.fp_wtp = 0;
        }

        let mut sme = 0usize;
        let mut cnt = 1usize;
        let mut level = 1;
        let mut next = cnt;
        self.region_find_index[0] = rend;
        self.regions[rend as usize].fp_level = 0;
        self.regions[rend as usize].fp_wtp = 0;

        let mut ok = false;

        while sme < cnt {
            let ri = self.region_find_index[sme] as usize;
            for i in 0..self.regions[ri].near.len() {
                if self.regions[ri].near_move[i] & mm != 0 {
                    continue;
                }
                let nr = self.regions[ri].near[i];
                let nru = nr as usize;
                let wtp = self.regions[ri].fp_wtp + self.regions[nru].fp_wt;

                if self.regions[nru].fp_level < 0 || wtp < self.regions[nru].fp_wtp {
                    self.regions[nru].fp_level = level;
                    self.regions[nru].fp_wtp = wtp;

                    if cnt >= self.region_find_index.len() {
                        self.region_find_index.resize(cnt + 1, 0);
                    }
                    self.region_find_index[cnt] = nr;
                    cnt += 1;

                    if nr == rstart {
                        ok = true;
                    }
                }
            }

            sme += 1;
            if sme >= next {
                next = cnt;
                level += 1;
                if !ok && level > maxpath {
                    if err {
                        panic!("find_path_in_region_run: maxpath exceeded (C++ ERROR_E)");
                    }
                    return 0;
                }
            }
        }

        if !ok {
            return 0;
        }

        let mut cnt = 1usize;
        let mut cr = rstart;
        let mut curwt = self.regions[cr as usize].fp_wtp;
        if let Some(p) = path.as_deref_mut() {
            p[0] = rstart;
        }

        loop {
            let ri = cr as usize;
            let mut nr_: i32 = -1;
            let mut wtp_ = curwt;
            for i in 0..self.regions[ri].near.len() {
                if self.regions[ri].near_move[i] & mm != 0 {
                    continue;
                }
                let nr = self.regions[ri].near[i];
                let wtp = self.regions[nr as usize].fp_wtp;
                if nr == rend {
                    nr_ = nr;
                    wtp_ = wtp;
                    break;
                } else if wtp <= wtp_ {
                    nr_ = nr;
                    wtp_ = wtp;
                }
            }
            if nr_ < 0 {
                panic!("find_path_in_region_run: descend dead end (C++ ERROR_E)");
            }

            cr = nr_;
            curwt = wtp_;
            if cnt as i32 + 1 > maxpath {
                if err {
                    panic!("find_path_in_region_run: path exceeds maxpath (C++ ERROR_E)");
                }
                return 0;
            }
            if let Some(p) = path.as_deref_mut() {
                p[cnt] = cr;
            }
            cnt += 1;
            if cr == rend {
                break;
            }
        }

        cnt
    }

    // ── Zone-level pathfinder (CMatrixMapLogic) ─────────────────────

    /// `PrepareBuf` (MatrixLogic.cpp:139) — size the zone scratch
    /// buffers to max(zone count, place count).
    fn prepare_buf(&mut self) {
        let n = self.zones.len().max(self.places.len());
        if self.zone_index.len() < n {
            self.zone_index.resize(n, 0);
        }
        if self.zone_index_access.len() < n {
            self.zone_index_access.resize(n, 0);
        }
    }

    /// `SetZoneAccess` (MatrixLogic.cpp:1437).
    fn set_zone_access(zones: &mut [Zone], list: &[i32], value: bool) {
        for &z in list {
            zones[z as usize].access = value;
        }
    }

    /// Deviation from the C++: zone-path traversal also skips links
    /// narrower than the 4-cell robot footprint
    /// (`near_zone_connect_size`, authored map data the C++ loads but
    /// never reads at runtime). The C++ happily routes robots through
    /// 1-cell gaps that `FindLocalPath` can never walk at size 4 —
    /// whole columns then wedge forever at the gap (ATOLL upper path).
    fn link_blocked(&self, zi: usize, i: usize, mask: u8) -> bool {
        self.zones[zi].near_zone_move[i] & mask != 0
            || self.zones[zi].near_zone_connect_size[i] < 4
    }

    /// The marking walk shared by the "found road from end" and
    /// "found road from start" branches of FindPathInZone
    /// (MatrixLogic.cpp:1575-1614 and 1652-1691 — the two blocks are
    /// verbatim duplicates with different walk targets). Descends from
    /// `from` along decreasing fp_level toward `target`, flagging each
    /// step and its passable neighbors as accessible.
    #[allow(clippy::too_many_arguments)]
    fn walk_mark_access(
        &mut self,
        mask: u8,
        from: i32,
        target: i32,
        level: &mut i32,
        accesscnt: &mut usize,
    ) {
        let mut curzone = from;
        while curzone != target {
            let zi = curzone as usize;
            let mut ok = false;
            for i in 0..self.zones[zi].near_zone.len() {
                if self.link_blocked(zi, i, mask) {
                    continue;
                }
                let newzone = self.zones[zi].near_zone[i];
                let nz = newzone as usize;
                if self.zones[nz].fp_level <= 0 {
                    continue;
                }
                if self.zones[nz].fp_level < *level {
                    curzone = newzone;
                    *level = self.zones[nz].fp_level;

                    if !self.zones[nz].access {
                        self.zones[nz].access = true;
                        self.zone_index_access[*accesscnt] = newzone;
                        *accesscnt += 1;
                    }
                    let czi = curzone as usize;
                    for u in 0..self.zones[czi].near_zone.len() {
                        if self.zones[czi].near_zone_move[u] & mask != 0 {
                            continue;
                        }
                        let n2 = self.zones[czi].near_zone[u];
                        let n2u = n2 as usize;
                        if self.zones[n2u].move_mask & mask != 0 {
                            continue;
                        }
                        if !self.zones[n2u].access {
                            self.zones[n2u].access = true;
                            self.zone_index_access[*accesscnt] = n2;
                            *accesscnt += 1;
                        }
                    }
                    ok = true;
                    break;
                }
            }
            if !ok {
                panic!("find_path_in_zone: access walk dead end (C++ ERROR_E)");
            }
        }
    }

    /// The final path-building descend of FindPathInZone
    /// (MatrixLogic.cpp:1548-1571 and 1773-1793 are verbatim
    /// duplicates): from `zstart`, repeatedly step to the first
    /// passable neighbor with a lower positive fp_level until `zend`.
    fn build_path_descend(
        &mut self,
        mask: u8,
        zstart: i32,
        zend: i32,
        mut level: i32,
        path: &mut [i32],
    ) -> usize {
        path[0] = zstart;
        let mut cnt = 1usize;
        let mut curzone = zstart;

        while curzone != zend {
            let zi = curzone as usize;
            let mut found = false;
            for i in 0..self.zones[zi].near_zone.len() {
                if self.link_blocked(zi, i, mask) {
                    continue;
                }
                let newzone = self.zones[zi].near_zone[i];
                let nz = newzone as usize;
                if self.zones[nz].fp_level <= 0 {
                    continue;
                }
                if self.zones[nz].fp_level < level {
                    curzone = newzone;
                    path[cnt] = newzone;
                    cnt += 1;
                    level = self.zones[nz].fp_level;
                    found = true;
                    break;
                }
            }
            if !found {
                panic!("find_path_in_zone: path descend dead end (C++ ERROR_E)");
            }
        }
        cnt
    }

    /// `FindPathInZone` (MatrixLogic.cpp:1446) — THE zone-level
    /// pathfinder the robots use. `nsh` is the chassis index (the
    /// passability bit is `1<<nsh`); `route` optionally constrains
    /// the search to a corridor around list `routeno` of a road route
    /// (every zone of every route road plus 2 rings of passable
    /// neighbors is flagged accessible). Writes zone indices to
    /// `path` (must hold `zones.len()` entries) and returns the
    /// count, 0 if unreachable.
    ///
    /// Three waves, as in the original: end→(road corridor or start),
    /// start→corridor, then end→start through accessible zones only.
    /// All access flags raised during the call are rolled back via
    /// the `zone_index_access` scratch list before returning. The
    /// debug-only `test` visualization parameter is dropped.
    pub fn find_path_in_zone(
        &mut self,
        nsh: usize,
        zstart: i32,
        zend: i32,
        route: Option<(&RoadRoute, usize)>,
        path: &mut [i32],
    ) -> usize {
        if zstart == zend {
            return 0;
        }
        // The C++ indexes m_Zone[-1] (UB) when ZoneFindNear failed —
        // e.g. a MoveTo destination outside the map. No path is the
        // sane answer.
        if zstart < 0
            || zend < 0
            || zstart as usize >= self.zones.len()
            || zend as usize >= self.zones.len()
        {
            return 0;
        }

        self.prepare_buf();

        let mask = 1u8 << nsh;
        let zs = zstart as usize;
        let ze = zend as usize;

        // End in a directly adjacent zone — no chassis filter, as in
        // the original (but a footprint-narrow link falls through to
        // the waves, see `link_blocked`).
        for i in 0..self.zones[zs].near_zone.len() {
            if self.zones[zs].near_zone[i] == zend
                && self.zones[zs].near_zone_connect_size[i] >= 4
            {
                path[0] = zstart;
                path[1] = zend;
                return 2;
            }
        }

        let mut accesscnt = 0usize;

        // Flag the route corridor accessible (MatrixLogic.cpp:1469-1500).
        if let Some((route, routeno)) = route {
            let stride = self.crotches.len();
            for i in 1..route.header[routeno].cnt as usize {
                let ri = route.units[routeno * stride + i]
                    .road
                    .expect("route unit without road");
                for u in 0..self.roads[ri].zones.len() {
                    let z1 = self.roads[ri].zones[u] as usize;
                    if self.zones[z1].move_mask & mask != 0 {
                        continue;
                    }
                    if !self.zones[z1].access {
                        self.zones[z1].access = true;
                        self.zone_index_access[accesscnt] = z1 as i32;
                        accesscnt += 1;
                    }

                    for t in 0..self.zones[z1].near_zone.len() {
                        if self.zones[z1].near_zone_move[t] & mask != 0 {
                            continue;
                        }
                        let z2 = self.zones[z1].near_zone[t];
                        let z2u = z2 as usize;
                        if self.zones[z2u].move_mask & mask != 0 {
                            continue;
                        }
                        if !self.zones[z2u].access {
                            self.zones[z2u].access = true;
                            self.zone_index_access[accesscnt] = z2;
                            accesscnt += 1;
                        }

                        for k in 0..self.zones[z2u].near_zone.len() {
                            if self.zones[z2u].near_zone_move[k] & mask != 0 {
                                continue;
                            }
                            let z3 = self.zones[z2u].near_zone[k];
                            let z3u = z3 as usize;
                            if self.zones[z3u].move_mask & mask != 0 {
                                continue;
                            }
                            if !self.zones[z3u].access {
                                self.zones[z3u].access = true;
                                self.zone_index_access[accesscnt] = z3;
                                accesscnt += 1;
                            }
                        }
                    }
                }
            }
        }

        for z in self.zones.iter_mut() {
            z.fp_level = 0;
        }

        // Wave 1: from the end until we hit the road corridor or the
        // start zone (MatrixLogic.cpp:1509-1543).
        let mut sme = 0usize;
        let mut level = 1;
        let mut cnt = 1usize;
        self.zone_index[0] = zend;
        self.zones[ze].fp_level = level;
        level += 1;
        let mut next = cnt;

        let mut zonefindok = -1;
        while sme < cnt {
            let zi = self.zone_index[sme] as usize;
            for i in 0..self.zones[zi].near_zone.len() {
                if self.link_blocked(zi, i, mask) {
                    continue;
                }
                let newzone = self.zones[zi].near_zone[i];
                let nz = newzone as usize;
                if newzone == zstart || self.zones[nz].access {
                    zonefindok = newzone;
                    sme = cnt;
                    break;
                }
                if self.zones[nz].fp_level != 0 {
                    continue;
                }
                if self.zones[nz].move_mask & mask != 0 {
                    continue;
                }

                self.zone_index[cnt] = newzone;
                cnt += 1;
                self.zones[nz].fp_level = level;
            }

            sme += 1;
            if sme >= next {
                next = cnt;
                level += 1;
            }
        }

        if zonefindok < 0 {
            // No path at all.
            Self::set_zone_access(&mut self.zones, &self.zone_index_access[..accesscnt], false);
            return 0;
        } else if zonefindok == zstart {
            // Direct path without touching the corridor.
            let pcnt = self.build_path_descend(mask, zstart, zend, level, path);
            Self::set_zone_access(&mut self.zones, &self.zone_index_access[..accesscnt], false);
            return pcnt;
        } else {
            // Reached the corridor: mark the walked stretch (and its
            // neighbors) accessible so wave 3 can pass through it.
            self.walk_mark_access(mask, zonefindok, zend, &mut level, &mut accesscnt);
        }

        // Wave 2: from the start until we hit an accessible zone
        // (MatrixLogic.cpp:1617-1649).
        for i in 0..cnt {
            self.zones[self.zone_index[i] as usize].fp_level = 0;
        }
        sme = 0;
        level = 1;
        cnt = 1;
        self.zone_index[0] = zstart;
        self.zones[zs].fp_level = level;
        level += 1;
        next = cnt;

        zonefindok = -1;
        while sme < cnt {
            let zi = self.zone_index[sme] as usize;
            for i in 0..self.zones[zi].near_zone.len() {
                if self.link_blocked(zi, i, mask) {
                    continue;
                }
                let newzone = self.zones[zi].near_zone[i];
                let nz = newzone as usize;
                if self.zones[nz].access {
                    zonefindok = newzone;
                    sme = cnt;
                    break;
                }
                if self.zones[nz].fp_level != 0 {
                    continue;
                }
                if self.zones[nz].move_mask & mask != 0 {
                    continue;
                }

                self.zone_index[cnt] = newzone;
                cnt += 1;
                self.zones[nz].fp_level = level;
            }

            sme += 1;
            if sme >= next {
                next = cnt;
                level += 1;
            }
        }

        if zonefindok < 0 {
            Self::set_zone_access(&mut self.zones, &self.zone_index_access[..accesscnt], false);
            return 0;
        } else {
            self.walk_mark_access(mask, zonefindok, zstart, &mut level, &mut accesscnt);
        }

        // Wave 3: end→start strictly through accessible zones
        // (MatrixLogic.cpp:1734-1766).
        for i in 0..cnt {
            self.zones[self.zone_index[i] as usize].fp_level = 0;
        }
        sme = 0;
        level = 1;
        cnt = 1;
        self.zone_index[0] = zend;
        self.zones[ze].fp_level = level;
        level += 1;
        next = cnt;

        zonefindok = -1;
        while sme < cnt {
            let zi = self.zone_index[sme] as usize;
            for i in 0..self.zones[zi].near_zone.len() {
                if self.link_blocked(zi, i, mask) {
                    continue;
                }
                let newzone = self.zones[zi].near_zone[i];
                let nz = newzone as usize;
                if newzone == zstart {
                    zonefindok = newzone;
                    sme = cnt;
                    break;
                }
                if self.zones[nz].fp_level != 0 {
                    continue;
                }
                if !self.zones[nz].access {
                    continue;
                }
                if self.zones[nz].move_mask & mask != 0 {
                    continue;
                }

                self.zone_index[cnt] = newzone;
                cnt += 1;
                self.zones[nz].fp_level = level;
            }

            sme += 1;
            if sme >= next {
                next = cnt;
                level += 1;
            }
        }

        let pcnt = if zonefindok < 0 {
            Self::set_zone_access(&mut self.zones, &self.zone_index_access[..accesscnt], false);
            return 0;
        } else {
            self.build_path_descend(mask, zstart, zend, level, path)
        };

        Self::set_zone_access(&mut self.zones, &self.zone_index_access[..accesscnt], false);

        pcnt
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test-only writer mirroring the `Save` layout
    /// (MatrixRoadNetwork.cpp:2453).
    struct W(Vec<u8>);

    impl W {
        fn new() -> Self {
            W(Vec::new())
        }
        fn int(&mut self, v: i32) {
            self.0.extend_from_slice(&v.to_le_bytes());
        }
        fn dword(&mut self, v: u32) {
            self.0.extend_from_slice(&v.to_le_bytes());
        }
        fn byte(&mut self, v: u8) {
            self.0.push(v);
        }
        fn bool_(&mut self, v: bool) {
            self.0.push(v as u8);
        }
        fn point(&mut self, x: i32, y: i32) {
            self.int(x);
            self.int(y);
        }
        fn rect(&mut self, l: i32, t: i32, r: i32, b: i32) {
            self.int(l);
            self.int(t);
            self.int(r);
            self.int(b);
        }
    }

    /// Builds a ver-27 blob (without the leading version dword):
    /// 2 zones, 2 crotches, 1 road, 2 places, 2 regions.
    fn build_blob() -> Vec<u8> {
        let mut w = W::new();

        // Zones.
        w.int(2);
        // zone 0
        w.point(5, 6);
        w.bool_(false); // bottleneck
        w.int(10); // size
        w.int(8); // perim
        w.rect(1, 2, 3, 4);
        w.bool_(true); // road
        w.byte(0); // move
        w.byte(1); // near cnt
        w.int(1); // near zone
        w.int(3); // connect size
        w.byte(2); // near move
        w.int(0); // region
        w.byte(1); // place cnt
        w.int(1); // place
        // zone 1
        w.point(15, 16);
        w.bool_(true);
        w.int(20);
        w.int(12);
        w.rect(5, 6, 7, 8);
        w.bool_(true);
        w.byte(4);
        w.byte(1);
        w.int(0);
        w.int(3);
        w.byte(0);
        w.int(1);
        w.byte(0);

        // Crotches.
        w.int(2);
        w.point(5, 6);
        w.int(0); // zone
        w.bool_(true); // critical
        w.int(0); // island
        w.point(15, 16);
        w.int(1);
        w.bool_(false);
        w.int(0);

        // Roads.
        w.int(1);
        w.int(0); // start crotch
        w.int(1); // end crotch
        w.int(2); // zone cnt
        w.int(0);
        w.int(1);
        w.int(14); // dist
        w.byte(0); // move

        // Places.
        w.int(2);
        // place 0
        w.point(3, 4);
        w.byte(1); // move
        w.byte(1); // border l
        w.byte(2); // border t
        w.byte(3); // border r
        w.byte(4); // border b
        w.byte(5); // border cnt
        w.byte(6); // edge l
        w.byte(7); // edge t
        w.byte(8); // edge r
        w.byte(9); // edge b
        w.byte(10); // edge cnt
        for u in 0..9 {
            w.bool_(u == 4);
        }
        w.byte(0); // cannon
        w.byte(1); // near cnt
        w.int(1);
        w.byte(2);
        // place 1 (cannon → move forced to 31)
        w.point(70, 4);
        w.byte(0);
        for _ in 0..10 {
            w.byte(0);
        }
        for _ in 0..9 {
            w.bool_(false);
        }
        w.byte(1); // cannon
        w.byte(0); // near cnt

        // Regions.
        w.int(2);
        // region 0
        w.point(5, 6);
        w.int(7); // radius place
        w.byte(1); // crotch cnt
        w.int(0);
        w.byte(1); // place cnt
        w.int(0);
        w.byte(2); // place all cnt
        w.int(0);
        w.int(1);
        w.byte(1); // near cnt
        w.int(1);
        w.byte(0);
        w.dword(0xAABBCCDD);
        // region 1
        w.point(15, 16);
        w.int(9);
        w.byte(1);
        w.int(1);
        w.byte(1);
        w.int(1);
        w.byte(0);
        w.byte(1);
        w.int(0);
        w.byte(4);
        w.dword(1);

        w.0
    }

    #[test]
    fn load_round_trip() {
        let blob = build_blob();
        let mut rn = RoadNetwork::new();
        rn.load(&blob).unwrap();

        // Zones.
        assert_eq!(rn.zones.len(), 2);
        let z0 = &rn.zones[0];
        assert_eq!(z0.center, Point::new(5, 6));
        assert!(!z0.bottleneck);
        assert_eq!(z0.size, 10);
        assert_eq!(z0.perim, 8);
        assert_eq!(
            z0.rect,
            Rect {
                left: 1,
                top: 2,
                right: 3,
                bottom: 4
            }
        );
        assert!(z0.road);
        assert_eq!(z0.move_mask, 0);
        assert_eq!(z0.near_zone, vec![1]);
        assert_eq!(z0.near_zone_connect_size, vec![3]);
        assert_eq!(z0.near_zone_move, vec![2]);
        assert_eq!(z0.region, 0);
        assert_eq!(z0.place, vec![1]);
        assert!(!z0.access);
        let z1 = &rn.zones[1];
        assert!(z1.bottleneck);
        assert_eq!(z1.move_mask, 4);
        assert_eq!(z1.region, 1);
        assert!(z1.place.is_empty());

        // Crotches: region copied from owning zone.
        assert_eq!(rn.crotches.len(), 2);
        assert_eq!(rn.crotches[0].zone, 0);
        assert!(rn.crotches[0].critical);
        assert_eq!(rn.crotches[0].region, 0);
        assert_eq!(rn.crotches[1].region, 1);
        assert_eq!(rn.crotches[0].roads, vec![0]);
        assert_eq!(rn.crotches[1].roads, vec![0]);

        // Road.
        assert_eq!(rn.roads.len(), 1);
        let r = &rn.roads[0];
        assert_eq!((r.start, r.end), (0, 1));
        assert_eq!(r.zones, vec![0, 1]);
        assert_eq!(r.dist, 14);
        assert_eq!(r.move_mask, 0);

        // Places.
        assert_eq!(rn.places.len(), 2);
        assert_eq!(rn.place_empty, -1);
        let p0 = &rn.places[0];
        assert_eq!(p0.pos, Point::new(3, 4));
        assert_eq!(p0.move_mask, 1);
        assert_eq!(
            (
                p0.border_left,
                p0.border_top,
                p0.border_right,
                p0.border_bottom,
                p0.border_cnt
            ),
            (1, 2, 3, 4, 5)
        );
        assert_eq!(
            (
                p0.edge_left,
                p0.edge_top,
                p0.edge_right,
                p0.edge_bottom,
                p0.edge_cnt
            ),
            (6, 7, 8, 9, 10)
        );
        let mut blockade = [false; 9];
        blockade[4] = true;
        assert_eq!(p0.blockade, blockade);
        assert_eq!(p0.cannon, 0);
        assert_eq!(p0.near, vec![1]);
        assert_eq!(p0.near_move, vec![2]);
        assert_eq!(p0.region, 0); // set by region 0's place list
        assert!(!p0.empty);
        assert_eq!(p0.empty_next, -1);
        // Cannon place gets all chassis bits.
        let p1 = &rn.places[1];
        assert_eq!(p1.cannon, 1);
        assert_eq!(p1.move_mask, 31);
        assert_eq!(p1.region, 1);

        // Regions.
        assert_eq!(rn.regions.len(), 2);
        let r0 = &rn.regions[0];
        assert_eq!(r0.center, Point::new(5, 6));
        assert_eq!(r0.radius_place, 7);
        assert_eq!(r0.crotch, vec![0]);
        assert_eq!(r0.place, vec![0]);
        assert_eq!(r0.place_all, vec![0, 1]);
        assert_eq!(r0.near, vec![1]);
        assert_eq!(r0.near_move, vec![0]);
        assert_eq!(r0.color, 0xAABBCCDD);
        assert_eq!(rn.regions[1].near_move, vec![4]);

        // Whole blob consumed exactly.
        let mut rn2 = RoadNetwork::new();
        assert!(rn2.load(&blob[..blob.len() - 1]).is_err());
    }

    #[test]
    fn init_pl_buckets_and_find_in_pl() {
        let mut rn = RoadNetwork::new();
        rn.load(&build_blob()).unwrap();
        rn.init_pl(128, 128);

        assert_eq!((rn.pl_size_x, rn.pl_size_y), (2, 2));
        // place 0 at (3,4) → bucket (0,0); place 1 at (70,4) → (1,0).
        assert_eq!(rn.pl_list[0].sme, 0);
        assert_eq!(rn.pl_list[0].cnt, 1);
        assert_eq!(rn.pl_list[1].sme, 1);
        assert_eq!(rn.pl_list[1].cnt, 1);
        assert_eq!(rn.pl_list[2].cnt, 0);

        assert_eq!(rn.find_in_pl(&Point::new(3, 4)), 0);
        assert_eq!(rn.find_in_pl(&Point::new(72, 6)), 1);
        assert_eq!(rn.find_in_pl(&Point::new(40, 40)), -1);

        // Index remap kept references intact (identity here).
        assert_eq!(rn.regions[0].place, vec![0]);
        assert_eq!(rn.regions[1].place, vec![1]);
        assert_eq!(rn.zones[0].place, vec![1]);
        assert_eq!(rn.places[0].near, vec![1]);
    }

    #[test]
    fn place_free_list() {
        let mut rn = RoadNetwork::new();
        rn.set_place_cnt(3);
        // Free list built back-to-front: head = 2 → 1 → 0.
        assert_eq!(rn.place_empty, 2);
        assert_eq!(rn.alloc_place(), 2);
        assert_eq!(rn.alloc_place(), 1);
        rn.free_place(2);
        assert_eq!(rn.place_empty, 2);
        assert_eq!(rn.alloc_place(), 2);
        assert_eq!(rn.alloc_place(), 0);
        // Exhausted → grows by 256.
        assert_eq!(rn.place_empty, -1);
        let no = rn.alloc_place();
        assert_eq!(rn.places.len(), 3 + 256);
        assert_eq!(no, 3 + 256 - 1);
    }

    /// 3-zone chain 0—1—2, no roads: wave 1 finds zstart directly and
    /// the descend produces the full path.
    #[test]
    fn find_path_in_zone_chain() {
        let mut rn = RoadNetwork::new();
        let mut zones = vec![Zone::default(); 3];
        zones[0].near_zone = vec![1];
        zones[0].near_zone_move = vec![0];
        zones[0].near_zone_connect_size = vec![5];
        zones[1].near_zone = vec![0, 2];
        zones[1].near_zone_move = vec![0, 0];
        zones[1].near_zone_connect_size = vec![5, 5];
        zones[2].near_zone = vec![1];
        zones[2].near_zone_move = vec![0];
        zones[2].near_zone_connect_size = vec![5];
        rn.zones = zones;

        let mut path = [0i32; 3];
        let cnt = rn.find_path_in_zone(0, 0, 2, None, &mut path);
        assert_eq!(cnt, 3);
        assert_eq!(&path[..3], &[0, 1, 2]);
        // Access flags rolled back.
        assert!(rn.zones.iter().all(|z| !z.access));

        // Adjacent zones short-circuit to a 2-entry path.
        let cnt = rn.find_path_in_zone(0, 0, 1, None, &mut path);
        assert_eq!(cnt, 2);
        assert_eq!(&path[..2], &[0, 1]);

        // A link narrower than the 4-cell footprint is not traversable
        // (the ATOLL upper-path wedge).
        rn.zones[1].near_zone_connect_size = vec![5, 1];
        rn.zones[2].near_zone_connect_size = vec![1];
        let cnt = rn.find_path_in_zone(0, 0, 2, None, &mut path);
        assert_eq!(cnt, 0);
        rn.zones[1].near_zone_connect_size = vec![5, 5];
        rn.zones[2].near_zone_connect_size = vec![5];

        // Chassis 1 blocked through zone 1 → no path.
        rn.zones[1].move_mask = 2;
        rn.zones[0].near_zone_move = vec![2];
        rn.zones[2].near_zone_move = vec![2];
        let cnt = rn.find_path_in_zone(1, 0, 2, None, &mut path);
        assert_eq!(cnt, 0);
    }

    /// Route cache: calc_route never sets start/end (same as the C++,
    /// where AddRoute's LIST_DEL no-op keeps the list empty), so
    /// find_route never matches and get_route always recalculates.
    #[test]
    fn route_cache_never_hits() {
        let mut rn = RoadNetwork::new();
        rn.load(&build_blob()).unwrap();

        let r1 = rn.get_route(0, 1);
        let r2 = rn.get_route(0, 1);
        assert_ne!(r1, r2);
        assert_eq!(rn.routes.len(), 2);

        // The single road 0→1 yields one completed list: the start
        // unit (no road) plus the [road0/crotch1] hop; the working
        // list is discarded by CalcRoute's final m_ListCnt--.
        let route = &rn.routes[r1];
        assert_eq!(route.list_cnt, 1);
        assert_eq!(route.header[0].cnt, 2);
        assert_eq!(route.header[0].dist, 14);
        assert_eq!(route.units[0].road, None);
        assert_eq!(route.units[0].crotch, 0);
        assert_eq!(route.units[1].road, Some(0));
        assert_eq!(route.units[1].crotch, 1);
    }
}
