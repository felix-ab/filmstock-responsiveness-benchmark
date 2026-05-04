# Filmstock Benchmark Prompt Blocks

These blocks are written to avoid ranges, slashes, bracketed choices, and alternative phrasing inside the generation prompt. Each rendered prompt should use exactly one scene block, one film stock block, one light source block, and one development scan block.

## Prompt Shell

```text
Square photorealistic editorial 35mm photograph of SCENE_BLOCK. Natural human proportions, realistic skin texture, art-directed but not glossy. Camera at eye level, 50mm lens, medium shot, subject clearly visible. FILM_STOCK_BLOCK. LIGHT_SOURCE_BLOCK. DEVELOPMENT_SCAN_BLOCK. No text, no logos, no watermark, no artificial digital smoothness.
```

## Scene Blocks

```text
a 25-year-old woman with short dark hair wearing an oversized red leather jacket and black jeans, seated on the floor of a lived-in city apartment, vinyl records, a chrome floor lamp, and a half-open window in the background
```

```text
a 22-year-old man wearing a faded denim jacket and white tank top, sitting on a worn velvet couch in a small backstage green room, an electric guitar, coiled cables, a makeup mirror, and posters without readable text in the background
```

```text
a 30-year-old woman wearing a black trench coat over a white dress, standing outside a corner store with glass doors, chrome fixtures, parked cars, and layered reflections in the background
```

## Film Stock Blocks

```text
Shot on Kodak Portra 400, 35mm color negative film, natural skin tones, soft pastel color response, gentle highlight rolloff
```

```text
Shot on CineStill 800T, 35mm tungsten-balanced color negative film, cinematic color separation, pronounced red halation around bright highlights when present
```

## Light Source Blocks

```text
Cool diffuse ambient light, soft open shadows, no dominant warm practical light source
```

```text
Warm tungsten practical lights with subtle neon spill, visible small bright light sources in the frame, deeper localized shadows
```

## Development Scan Blocks

```text
Rated at box speed, clean neutral Frontier lab scan, restrained contrast, fine natural grain
```

```text
Pushed one stop in development, stronger contrast, denser shadows, coarser visible grain, slight color shift
```

## File Naming

Use a compact deterministic token:

```text
model_scene_film_light_scan_rep
```

Example:

```text
chatgpt_image_2_apartment_portra_cool_clean_rep01
```
