
    docker build -f fu_pl . -f Dockerfile


    $docker build -t chokj_fu . -f Dockerfile

    $docker run -ti \
    --gpus all --name chokj_pl \
    -v /home/chokj/mimic:/workspace/datasets/open_dataset/mimic \
    -v /mnt/nas252/CXR_PreProcessed_Data/open_dataset:/workspace/datasets/open_dataset \
    -v /home/chokj/nas252_cxr_fu:/workspace/datasets/nas252 \
    -v /home/chokj/nas125_cxr_fu:/workspace/datasets/nas125 
    -v /mnt:/mnt \
    -v /home/chokj/nas43:/workspace/datasets/nas43 \
    --shm-size=32G -p 9873:9873 fu_pl:latest /bin/bash