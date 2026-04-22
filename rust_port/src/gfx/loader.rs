/// Load raw bytes from a file path.
/// On native: reads from filesystem via tokio.
/// On WASM: fetches via the browser fetch API.
pub async fn load_bytes(path: &str) -> anyhow::Result<Vec<u8>> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        Ok(tokio::fs::read(path).await?)
    }

    #[cfg(target_arch = "wasm32")]
    {
        use wasm_bindgen::JsCast;
        use wasm_bindgen_futures::JsFuture;
        use web_sys::{Request, RequestInit, Response};

        let opts = RequestInit::new();
        let request =
            Request::new_with_str_and_init(path, &opts).map_err(|e| anyhow::anyhow!("{e:?}"))?;

        let window = web_sys::window().unwrap();
        let response = JsFuture::from(window.fetch_with_request(&request))
            .await
            .map_err(|e| anyhow::anyhow!("{e:?}"))?;

        let response: Response = response.dyn_into().unwrap();
        if !response.ok() {
            anyhow::bail!(
                "fetch {}: HTTP {} {}",
                path,
                response.status(),
                response.status_text()
            );
        }
        let buffer = JsFuture::from(response.array_buffer().unwrap())
            .await
            .map_err(|e| anyhow::anyhow!("{e:?}"))?;

        Ok(js_sys::Uint8Array::new(&buffer).to_vec())
    }
}
