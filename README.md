# Video Copy Detection (Updating)
Dự án NCKH của nhóm các bạn sinh viên DSEB64A

Topic: Video Matching

Motivation:
The booming of GenAI techniques has been opening up a new horizon for content creation. With the
help of GenAI applications, text, image, video music are created faster with high quality than ever.
Our project is to tackle a specific problem in video matching. Given a short clip (a news clip registered by
a journalist for instance), we are interested in deploying an Information Retrieval Engine to find
relevant, similar videos in given video database (of a news broadcasting company). These resulting
videos could be used for creating official news videos related to the content of the inputted clip.

Tasks:
Input: Dataset from MetaAI

**In this project, we will do two phases:**

- **First: Check if a video is copy or not**.
In this task, we will extract video into frames, then embedding these frames used CLIP model.
After that, to reduce the amount of calculation cost, we will check if a video is edited or not because we see that if a video is edited -> it is usually a copied video by let the embeddings into a Roberta Model.
After checking a video is edited, or not. If unedited -> convert it to a random vector. If edited -> extract by swinv2 model -> We can learn the embeddings by contrastive learning. 
In inference phase, we extract both query videos and reference videos into embedding -> We use KNN to get the top equivalent features.


****- Second: If the video is copied -> what segment? ( Updating )**
