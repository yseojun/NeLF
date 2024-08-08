// main.js
import { updateSceneWithInput} from './socketConnection.js';

async function animate() {
	await updateSceneWithInput();
	requestAnimationFrame(animate);
}

animate();