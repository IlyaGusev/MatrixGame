//! Platform selection for the sound backend (SOUND_PROPOSAL.md §3).
//! WASM gets the WebAudio graph; native stays silent (the native
//! binary is a dev tool — rodio is an explicit non-goal).

use crate::matrix_game::sound::SoundOutput;

pub fn make_output() -> Box<dyn SoundOutput> {
    #[cfg(target_arch = "wasm32")]
    {
        Box::new(super::audio_web::WebOutput::new())
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        Box::new(crate::matrix_game::sound::NullOutput)
    }
}
