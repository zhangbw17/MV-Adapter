# Failure cases of MV-Adapter

We summarize two failure modes of our method from comprehensive case studies: 
- Since we use a fixed number of frames per video in training, our method may fail to capture fine movements in long videos as the differences between frames aggregate.
- Results may be incorrect when the caption is related to the audio contents.

Examples of these two modes are put into ./long_video and ./audio directories respectively. Each directory contains one retrieval result, including caption.txt (the query we use to search), gt_\*.mp4 and pred_\*.mp4 (the groundtruth and predicted video, where * represents the index number of the video in MSR-VTT.)


To be more specific, we will go over each example:
- **long_video_0.** In this case, the clips featuring "people" are quite concentrated and "fade" quickly. Given the video's duration of 27 seconds, the long intervals between extracted frames result in key information being omitted, making it impossible to match the description. Therefore, another video, where the fading effect and the characters are clearer, is returned instead.

- **long_video_1.** The groundtruth video is long and the shot that corresponds to the target in the query (walking down a short runway) is relatively short. Since the number of input frames is fixed, non-target information in videos tends to dominate the input, making the model fail to parse out the fine movement ("walking" and "short runway") from the groundtruth. Eventually, the model returns a similar video that contains "walking" but on a "long runway" (should be a short runway).

- **audio_0.** Audio information is necessary in order to determine the topic of talking.

- **audio_1.** Though the retrieved result is visually similar to groundtruth, the contents of the talk do not match that of the query text. With the help of audio content (like transcripts from ASR), the results can be corrected.