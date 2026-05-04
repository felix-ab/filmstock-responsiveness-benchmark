# Filmstock Responsiveness Benchmark

This repository is a small statistical proof of concept for measuring how image generation models respond to filmstock, lighting, and development-scan language. The motivating idea is simple: prompt adherence benchmarks are necessary, but they usually emphasize object correctness, compositional relationships, text rendering, and realism. For creative work, that is not the whole story. A useful image model should also move in the intended visual direction when the prompt says *Kodak Portra 400*, *CineStill 800T*, *warm tungsten practicals*, or *pushed one stop in development*.

The study compares **ChatGPT Image 2** and **Grok Imagine** on a controlled full-factorial prompt set:

- 3 photographic scenes
- 2 filmstock conditions: Kodak Portra 400 and CineStill 800T
- 2 lighting conditions: cool diffuse ambient and warm tungsten practical
- 2 development-scan conditions: clean neutral scan and pushed scan
- 2 models
- 2 within-run replicates
- 3 independent runs

That produces **288 normalized 1024 by 1024 images**, **195 extracted image features**, and **432 blocked prompt-response contrast rows**.

## Why This Exists

The project is framed as a “filmmaker’s benchmark” seed rather than a finished leaderboard. Current text-to-image evaluations are strong at asking whether a model included the requested objects, counted correctly, followed compositional instructions, or produced an aesthetically preferred image. This benchmark asks a complementary question: when the cue is specifically photographic or cinematic, can the response be measured in color, tone, texture, spatial color covariance, and feature-space movement?

The main takeaway is not “one model wins.” The more useful result is methodological: artistic responsiveness is multidimensional. In this sample, lighting produced the strongest feature-space movement, model-by-lighting differences were robust, and filmstock/scan cues were detectable but subtler.

## Repository Layout

```text
paper/
  filmstock_responsiveness_benchmark.docx
  filmstock_responsiveness_benchmark.pdf

code/
  generate_full_dataset.py
  analyze_filmstock_dataset.py
  analyze_filmstock_pooled_dataset.py
  run_statistical_analysis_pooled.py
  run_advanced_statistical_suite.py
  generate_v2_figures.py
  build_final_writeup_v2.py
  table_geometry.py

data/
  images/
    run1/ ... run3/       # normalized 1024x1024 images used in analysis
  manifests/              # prompt, condition, model, and image metadata
  analysis/               # feature tables, PCA scores, tests, summaries

figures/
  paper_figures/          # final and appendix figure candidates

workflows/
  ComfyUI workflow/API JSON files for ChatGPT Image 2 and Grok Imagine

prompts/
  prompt_blocks.md
```

## Reproducing the Analysis

Create an environment and install the analysis dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Regenerate the pooled statistical outputs from the included analysis-ready CSVs:

```bash
python code/run_statistical_analysis_pooled.py
python code/run_advanced_statistical_suite.py
python code/generate_v2_figures.py
python code/build_final_writeup_v2.py
```

The scripts default to repo-relative paths. If you move the data, set `FILMSTOCK_ANALYSIS_DIR`, `FILMSTOCK_ASSET_DIR`, or `FILMSTOCK_OUTPUT_DOCX`.

## Regenerating Images

The original generations were run through ComfyUI with provider-specific custom nodes. The workflow JSON files are included as templates and require local provider authentication to run.

Typical local configuration:

```bash
export COMFYUI_URL=http://127.0.0.1:8288
export COMFYUI_ROOT=/path/to/ComfyUI
export FILMSTOCK_WORKFLOW_DIR=$PWD/workflows
export FILMSTOCK_OUTPUT_ROOT=/path/to/output
```

Then run:

```bash
python code/generate_full_dataset.py
```

Each full run generates 96 images. The public dataset contains three completed runs.

## Limitations

This is intentionally small. It is best read as a statistical and methodological prototype, not a final general-purpose benchmark. A larger version should include more models, human ratings, more cinematographic terms, more scenes, reference image sets, and thousands of prompt conditions.

## License

Code is MIT licensed. Data, figures, prompts, and writing are shared under CC BY 4.0 where permitted by model-provider terms. Generated images may also be subject to the relevant providers’ terms of use.
