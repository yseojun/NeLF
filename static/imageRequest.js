// imageRequest.js
import { socket } from './socketConnection.js';

export async function requestNewImage(x, y, z, roll, pitch) {
    socket.emit('request_new_image', { 'x': x, 'y': y, 'z': z, 'roll': roll, 'pitch': pitch });
}