/// Top-level game state: units, buildings, resources, map.
pub struct World {
    pub tick: u64,
    pub elapsed: f32,
}

impl World {
    pub fn new() -> Self {
        Self {
            tick: 0,
            elapsed: 0.0,
        }
    }

    pub fn update(&mut self, dt: f32) {
        self.tick += 1;
        self.elapsed += dt;
    }
}
