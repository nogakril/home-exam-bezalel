# Souvenir Snow Globe: Pyramid Blend Memories

*Material & Culture Visual Research Project — Image Processing & Emotional Memory (2023)*

A visual experiment exploring how cherished objects evoke memory and emotion. For this project, I blended images of treasured memories with photos of my vintage glass collection using **Pyramid Blending**, creating a dreamy “souvenir snow globe” visual effect.  
The resulting **stop-motion video** brings out the playful colors and textures of glass while honoring joyful past experiences.

---

##  Final Outcome  
Check out the full visual experience on YouTube:  
[▶ Watch “Souvenir Snow Globe” video](https://www.youtube.com/watch?v=JmxhgrFEc20)

---

##  Concept & Motivation
Inspired by the emotional resonance of everyday objects, this work asks: *How can a simple material—like glass—be transformed into a container of memory?*  
By layering imagery, I aimed to evoke nostalgia and personal narrative through a meditative, sensory visual.

---

##  Technical Approach
- **Image Processing Algorithm:** Used *Pyramid Blending*—a multi-resolution blending technique—on pairs of images (a memory + a glass object)
- **Visual Output:** Generated as a series of frames with layered transparency
- **Stop-Motion Assembly:** Combined results into a looping animation to evoke the snow-globe aesthetic

---

##  Project Structure
home-exam-bezalel/
├── main.py # Core workflow: load images, blend, export frames
├── pictures_generator.py # Helper for batch image processing tasks
└── README.md # Project overview (this file)

---

##  Usage / Reproduce Steps
*(Add instructions if you’d like to open source the code)*

1. Clone the repo:
   ```bash
   git clone https://github.com/nogakril/home-exam-bezalel.git
   cd home-exam-bezalel
2. Prepare image pairs:
   Place your memory-image and glass-object image in a folder (e.g., /input)
4. Run the blending script:
   ```bash
    python main.py --input /input --output /output
4. Use pictures_generator.py to batch process or generate stop-motion frames.
5. Compile frames into a video using your tool of choice (e.g., ffmpeg).
