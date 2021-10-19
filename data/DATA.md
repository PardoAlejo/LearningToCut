# Learning to Cut by Watching Movies 

The data from this project was downloaded from youtube.

# Original videos

To access the original videos check their youtube ID under the column `video_id` in the files: `subset_moviescenes_shotcuts_{train,val}.csv`.

Go to youtube an replace the id in this URL: `https://www.youtube.com/watch?v={video_id}`

# Feature Extraction

The features used for the paper are shared in the main README.md

However, if you wanna extract features yourself we follow the repos below.

| **Resource** | Features |  Notes |
| ----         |:-----:   | :-----: |
| **Audio** | [ResNet-18 VGGSound](https://github.com/PardoAlejo/VGGSoundFeatures.git) | Parameters used can be found in the repo|
| **Video** | [ResNexT-101 Kinetics](https://github.com/antoine77340/video_feature_extractor.git) | Snippet size: 16 Frames at 24FPS with 8-frame overlapping windows|
