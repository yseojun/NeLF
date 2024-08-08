// eventHandlers.js

import Camera from './Camera.js';

const camera = new Camera();

export let isKeyPressed = false;
export let isMouseMoving = false;

let isMousePressed = false;
let lastMouseEvent = null;

let lastKeyEvent = null;

export function handleKeyboardInput() {
    const step = 0.01;
    const key = lastKeyEvent.key;
    switch (key) {
        case 'w':
            camera.move_forward(step);
            break;
        case 's':
            camera.move_backward(step);
            break;
        case 'a':
            camera.move_left(step);
            break;
        case 'd':
            camera.move_right(step);
            break;
        default:
            return;
    }
    document.getElementById('x').value = camera.position[1];
    document.getElementById('y').value = camera.position[2];
    document.getElementById('z').value = camera.position[0];
}

export function handleKeyboardAndMouseInput() {
    const step = 0.01;
    const key = lastKeyEvent.key;
    switch (key) {
        case 'w':
            camera.move_forward(step);
            break;
        case 's':
            camera.move_backward(step);
            break;
        case 'a':
            camera.move_left(step);
            break;
        case 'd':
            camera.move_right(step);
            break;
        default:
            return;
    }
    const mouseStep = 0.05;
    const dx = lastMouseEvent.movementX || lastMouseEvent.mozMovementX || lastMouseEvent.webkitMovementX || 0;
    const dy = lastMouseEvent.movementY || lastMouseEvent.mozMovementY || lastMouseEvent.webkitMovementY || 0;

    if (dx !== 0) {
        camera.rotate_left_right(dx * mouseStep); 
    }
    if (dy !== 0) {
        camera.rotate_up_down(dy * mouseStep);
    }

    document.getElementById('x').value = camera.position[0];
    document.getElementById('y').value = camera.position[1];
    document.getElementById('z').value = camera.position[2];
    document.getElementById('yaw').value = camera.rotation[0];
    document.getElementById('pitch').value = camera.rotation[1];
}

export function handleMouseMove() {
    const mouseStep = 0.05;
    const dx = lastMouseEvent.movementX || lastMouseEvent.mozMovementX || lastMouseEvent.webkitMovementX || 0;
    const dy = lastMouseEvent.movementY || lastMouseEvent.mozMovementY || lastMouseEvent.webkitMovementY || 0;
    
    if (dx !== 0) {
        camera.rotate_left_right(dx * mouseStep); 
    }
    if (dy !== 0) {
        camera.rotate_up_down(dy * mouseStep);
    }
    document.getElementById('yaw').value = camera.rotation[0];
    document.getElementById('pitch').value = camera.rotation[1];
}

document.addEventListener('keydown', (event) => {
    isKeyPressed = true;
    lastKeyEvent = event;
});

document.addEventListener('keyup', () => {
    isKeyPressed = false;
    lastKeyEvent = null;
});

document.addEventListener('mousedown', () => {
    isMousePressed = true;
});

document.addEventListener('mouseup', () => {
    isMousePressed = false;
    lastMouseEvent = null;
    isMouseMoving = false;
});

document.addEventListener('mousemove', (event) => {
    enableMouseMove(event);
});

function enableMouseMove(event) {
    if (isMousePressed) {
        isMouseMoving = true;
        lastMouseEvent = event;
    }
}