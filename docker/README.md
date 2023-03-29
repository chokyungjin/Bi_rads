
    docker build -f fu_pl . -f Dockerfile

    $docker run -ti \
    --gpus all --name chokj_pl \
    -v /mnt:/mnt \
    --shm-size=32G -p 9873:9873 fu_pl:latest /bin/bash