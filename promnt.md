You are a senior Python AI/ML engineer.

I want to build a LOCAL, self-hosted prototype for an AI cartoon story video generator.

IMPORTANT:
- No frontend.
- No backend/API server.
- No database.
- No Docker for now.
- No paid third-party APIs.
- No OpenAI API.
- No ElevenLabs API.
- No Google/AWS AI APIs.
- Everything must run locally.
- Use Python.
- Keep the prototype extremely simple.
- Do NOT build production architecture yet.
- Do NOT over-engineer it.

==================================================
GOAL
==================================================

I want to give the system a simple STORY as input.

Example:

"एक आलसी बिल्ली रोज़ दूध चुराने की कोशिश करती है,
लेकिन हर बार दुकानदार उसे पकड़ लेता है। आखिर में
बिल्ली एक मजेदार तरीका खोजती है।"

The system should automatically create a short cartoon-style video.

Initial target:

30-second video.

Later this can become:

1 minute
2 minutes
5 minutes
10 minutes

But ONLY build/test the 30-second version now.


==================================================
TARGET PIPELINE
==================================================

Story
  ↓
Local LLM
  ↓
Scene breakdown
  ↓
Scene descriptions
  ↓
Local cartoon image generation
  ↓
Hindi voice generation
  ↓
Simple image animation
  ↓
FFmpeg
  ↓
Final MP4


==================================================
IMPORTANT MVP LIMITATION
==================================================

DO NOT try to create advanced AI video animation yet.

For the first PoC:

Generate approximately 5–6 cartoon images.

Each image represents one scene.

Then create motion using:

- zoom in
- zoom out
- pan left
- pan right
- slight movement
- transitions

This is intentionally NOT full AI video generation.

The purpose is to test whether:

Story → Cartoon scenes → Hindi narration → Video

works locally with acceptable quality.


==================================================
STEP 1 — ENVIRONMENT CHECK
==================================================

First create:

check_environment.py

It must detect:

- OS
- Python version
- RAM
- NVIDIA GPU
- GPU model
- GPU VRAM
- CUDA availability
- PyTorch version
- FFmpeg
- available disk space

Print a clear report.

Example:

=============================
SYSTEM
=============================

OS:
Python:
RAM:

GPU:
VRAM:
CUDA:

PyTorch:
FFmpeg:

Disk:

=============================


IMPORTANT:

Do not install large models before checking the hardware.


==================================================
STEP 2 — STORY INPUT
==================================================

Create a simple:

story.txt

Example:

"एक आलसी बिल्ली रोज़ दूध चुराने की कोशिश करती है,
लेकिन हर बार दुकानदार उसे पकड़ लेता है।
एक दिन बिल्ली ने ऐसा प्लान बनाया कि दुकानदार भी
हैरान रह गया।"


The Python program should read this file.

Also support:

python main.py --story story.txt


==================================================
STEP 3 — LOCAL STORY/SCENE GENERATION
==================================================

Use a LOCAL LLM.

No cloud API.

The LLM should convert the story into approximately
5–6 scenes.

For example:

Scene 1:
A lazy cartoon cat sitting outside a milk shop.

Scene 2:
The cat looking at a glass of milk.

Scene 3:
The cat secretly approaching the shop.

Scene 4:
The shopkeeper catches the cat.

Scene 5:
The cat creates a funny plan.

Scene 6:
Funny ending.

The output should be structured JSON.

Example:

{
  "title": "...",
  "scenes": [
    {
      "scene": 1,
      "duration": 5,
      "narration_hindi": "...",
      "visual_prompt": "..."
    }
  ]
}

Save:

output/scenes.json


IMPORTANT:

The local LLM should NOT directly generate images.

It only creates:

- scene description
- Hindi narration
- visual prompt
- approximate duration


==================================================
STEP 4 — CHARACTER CONSISTENCY
==================================================

This is an important experiment.

The same main character should appear consistently
throughout the scenes.

For example:

Character:

"small orange cartoon cat,
large expressive eyes,
blue collar,
funny face,
2D animated cartoon style"

Create a reusable CHARACTER DESCRIPTION.

Every scene's image prompt must include this
character description.

Do NOT implement advanced character consistency
systems such as LoRA/IP-Adapter yet unless they are
very easy to add.

The first goal is simply to test whether prompt-based
consistency is acceptable.


==================================================
STEP 5 — LOCAL IMAGE GENERATION
==================================================

Use a LOCAL open-source image generation model.

No API.

Prefer a model that can realistically run on my
available NVIDIA GPU.

First inspect my GPU VRAM and then recommend the
appropriate model.

Possible approaches can include:

- Stable Diffusion
- SDXL
- Flux variant
- another lightweight local image model

Do NOT blindly download a huge model.

The model must generate approximately:

5–6 images.

Save:

output/images/scene_001.png
output/images/scene_002.png
...
output/images/scene_006.png


Each image should be:

cartoon / animated style.

IMPORTANT:

Do not aim for photorealism.


==================================================
STEP 6 — HINDI TTS
==================================================

Generate Hindi narration locally.

NO cloud TTS API.

Use a local Hindi-capable neural TTS model.

The voice should sound as natural as possible.

Initially use ONE narrator voice.

Do NOT implement multiple character voices yet.

Generate:

output/audio/scene_001.wav
output/audio/scene_002.wav
...

Also create:

output/audio_manifest.json

containing:

- scene
- text
- audio duration
- file path


==================================================
STEP 7 — SCENE TIMING
==================================================

Each scene should be approximately synchronized
with its narration.

Example:

Scene image:
5 seconds

Narration:
4.7 seconds

Keep image duration around narration duration.

If necessary, add a small padding.

Do not create complicated lip-sync.


==================================================
STEP 8 — SIMPLE ANIMATION
==================================================

This is VERY IMPORTANT.

Do NOT generate AI video clips yet.

Instead animate the generated images.

Implement simple camera movement:

Scene 1:
slow zoom in

Scene 2:
slow pan left

Scene 3:
slow zoom out

Scene 4:
slow pan right

Scene 5:
zoom in

Scene 6:
slight zoom + transition

Use FFmpeg or MoviePy.

The final result should feel like an animated
cartoon/story video rather than a slideshow.

Keep motion subtle.


==================================================
STEP 9 — SUBTITLES
==================================================

Generate Hindi subtitles from the narration.

Create:

output/subtitles.srt

Burn subtitles into the final video OR provide
them as a separate file.

For the PoC, burned-in subtitles are acceptable.


==================================================
STEP 10 — FINAL VIDEO
==================================================

Use FFmpeg to combine:

- animated scene images
- Hindi narration
- optional background music
- optional simple sound effects
- subtitles

Output:

output/final_video.mp4


Target:

30-second video.


==================================================
BACKGROUND MUSIC
==================================================

Do NOT download copyrighted music.

For the first PoC:

OPTION A:
No music.

OR

OPTION B:
Use a locally available royalty-free/test audio file.

Make it optional.

Do NOT build an automated music-generation system yet.


==================================================
SOUND EFFECTS
==================================================

Do NOT implement complex automatic SFX generation.

Optional:

Allow local WAV files such as:

door.wav
pop.wav
whoosh.wav

But this is NOT required for the first test.


==================================================
PROJECT STRUCTURE
==================================================

Keep it simple:

cartoon-video-poc/
│
├── story.txt
│
├── input/
│
├── output/
│   ├── images/
│   ├── audio/
│   ├── scenes.json
│   ├── subtitles.srt
│   └── final_video.mp4
│
├── models/
│
├── src/
│   ├── story.py
│   ├── image_generation.py
│   ├── tts.py
│   ├── animation.py
│   ├── subtitles.py
│   └── video.py
│
├── check_environment.py
├── main.py
├── requirements.txt
└── README.md


==================================================
CLI
==================================================

The final prototype should run like:

python main.py --story story.txt


Optional:

python main.py \
  --story story.txt \
  --duration 30


==================================================
PERFORMANCE REPORT
==================================================

At the end print:

========================================
AI CARTOON VIDEO POC
========================================

Story processing: XX sec

Scene generation: XX sec

Image generation:
Scene 1: XX sec
Scene 2: XX sec
Scene 3: XX sec
Scene 4: XX sec
Scene 5: XX sec
Scene 6: XX sec

Total image generation: XX sec

Hindi TTS: XX sec

Animation: XX sec

FFmpeg: XX sec

TOTAL PROCESSING TIME: XX sec

VIDEO LENGTH: XX sec

GPU: XXXXX
VRAM: XX GB

========================================


==================================================
FIRST TEST
==================================================

Do NOT immediately generate a 10-minute video.

First use a very simple story and generate:

30-second video
5–6 scenes
1 narrator
cartoon images
simple camera animation
Hindi narration
Hindi subtitles


==================================================
QUALITY CHECK
==================================================

After generating the video, evaluate:

1. Are the cartoon images visually good?
2. Is the same character reasonably consistent?
3. Does the story flow correctly?
4. Does Hindi narration sound natural?
5. Does narration match scenes?
6. Does the animation feel better than a slideshow?
7. Is the final video watchable?
8. How long did generation take?
9. How much GPU VRAM was used?
10. Did the GPU remain fully utilized?


==================================================
VERY IMPORTANT
==================================================

This is a FEASIBILITY PROTOTYPE.

Do NOT build:

- React
- Next.js
- FastAPI
- Express
- PostgreSQL
- Redis
- authentication
- user accounts
- payment system
- queues
- Docker
- cloud deployment
- multi-user support
- admin dashboard

Do NOT build these yet.

Only prove:

STORY
→ LOCAL LLM
→ SCENES
→ LOCAL CARTOON IMAGES
→ LOCAL HINDI TTS
→ SIMPLE ANIMATION
→ FFMPEG
→ 30 SECOND MP4


==================================================
DEVELOPMENT ORDER
==================================================

Follow this exact order:

1. Inspect hardware.
2. Check CUDA.
3. Check FFmpeg.
4. Recommend practical local models.
5. Test local LLM.
6. Test ONE image generation.
7. Test Hindi TTS.
8. Test FFmpeg animation.
9. Connect everything.
10. Generate one complete 30-second video.
11. Measure performance.
12. Only then suggest improvements.

If any model is too heavy for my GPU, DO NOT silently
replace it with a cloud API.

Explain the limitation and choose a smaller local
model.

The priority is:

LOCAL
→ SIMPLE
→ CHEAP
→ FAST
→ TESTABLE
→ REASONABLE QUALITY

NOT production quality yet.


My local hardware is:

CPU: AMD Ryzen 7 3700X (8 cores / 16 threads)
RAM: 64 GB
GPU: NVIDIA RTX 2060 8 GB VRAM

Optimize all model selection and inference settings for this hardware.

Prefer CUDA/GPU whenever supported.
Avoid models that realistically require more than 8 GB VRAM unless a memory-efficient/quantized/offloading configuration is available.