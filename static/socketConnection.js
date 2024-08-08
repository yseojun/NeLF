// socketConnection.js
import { updateImage } from './imageUpdate.js';
import { requestNewImage } from './imageRequest.js';
import { handleKeyboardInput, handleKeyboardAndMouseInput, handleMouseMove, isKeyPressed, isMouseMoving } from './eventHandlers.js';

let isRequestingImage = false;
let isDownloading = false;

const socket = io.connect(`http://203.246.87.66:6006`);

socket.on('new_image', async (data) => {
    isDownloading = true;
    await updateImage(data);
    isRequestingImage = false;
    isDownloading = false;
});

socket.on('size', (data) => {
    document.getElementById('depth').textContent = data.depth;
    document.getElementById('width').textContent = data.width;
});

export async function updateSceneWithInput(){
    let shouldRequestNewImage = false;

    if (!isRequestingImage && !isDownloading) {
        performance.mark('start');
        if (isKeyPressed && isMouseMoving) {
            handleKeyboardAndMouseInput();
            shouldRequestNewImage = true;
        } else if (isKeyPressed) {
            handleKeyboardInput();
            shouldRequestNewImage = true;
        } else if (isMouseMoving) {
            handleMouseMove();
            shouldRequestNewImage = true;
        }
		
        if (shouldRequestNewImage) {
            var x = document.getElementById('x').value;
            var y = document.getElementById('y').value;
            var z = document.getElementById('z').value;
            var yaw = document.getElementById('yaw').value;
            var pitch = document.getElementById('pitch').value;
            requestNewImage(x, y, z, yaw, pitch);
            isRequestingImage = true;
        }
    }
}

export { socket };