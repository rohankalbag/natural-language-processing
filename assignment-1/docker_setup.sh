# to enable this script : chmod +x docker_setup.sh
# usage : ./docker_setup.sh

IP=$(ipconfig getifaddr en0)
X11_PORT=0
HOST_PATH=~/Repositories/cs772-iitb
DOCKER_IMAGE=rohankalbag/tsetlin:latest

echo "Running docker container"
echo "HOST_IP: "$IP":"$X11_PORT


docker run --rm -it --platform linux/amd64 -v $HOST_PATH:/root -e DISPLAY=$IP:$X11_PORT $DOCKER_IMAGE