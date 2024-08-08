const X=0;
const Y=1;
const Z=2;
const YAW=0;
const PITCH=1;

export default class Camera {
    constructor(x = 0, y = 0, z = 0, yaw = 0, pitch = 0, roll = Math.PI) {
        this.position = [x, y, z];
        this.rotation = [yaw, pitch, roll];
        this.direction_vector = this.calculate_direction_vector();
        this.last_pos = null;
    }

    calculate_direction_vector() {
        var yaw = this.rotation[0];
        var pitch = this.rotation[1];
        var directionVector = [
            Math.cos(yaw) * Math.cos(pitch),
            Math.sin(yaw),
            Math.cos(yaw) * Math.sin(pitch)
        ];
        var norm = Math.sqrt(directionVector.reduce(function(acc, val) {
            return acc + Math.pow(val, 2);
        }, 0));
        if (norm > 0) {
            directionVector = directionVector.map(function(val) {
                return val / norm;
            });
        }
        return directionVector;
    }

    move_forward(distance) {
        const [dx, dy, dz] = this.direction_vector;
        this.position[X] -= dx * distance;
        this.position[Y] -= dy * distance;
        this.position[Z] -= dz * distance;
    }

    move_backward(distance) {
        const [dx, dy, dz] = this.direction_vector;
        this.position[X] += dx * distance;
        this.position[Y] += dy * distance;
        this.position[Z] += dz * distance;
    }

    move_left(distance) {
        const [dx, dy, dz] = this.direction_vector;
        const left_vector = [
            dz * Math.sin(Math.PI / 2) - dy * Math.cos(Math.PI / 2),
            dx * Math.cos(Math.PI / 2) - dz * Math.sin(Math.PI / 2),
            dy * Math.sin(Math.PI / 2) - dx * Math.cos(Math.PI / 2)
        ];
        const norm = Math.sqrt(left_vector.reduce((acc, val) => acc + val * val, 0));
        const normalized_left_vector = norm > 0 ? left_vector.map(val => val / norm) : left_vector;
        this.position[X] -= normalized_left_vector[0] * distance;
        this.position[Y] -= normalized_left_vector[1] * distance;
        this.position[Z] -= normalized_left_vector[2] * distance;
    }

    move_right(distance) {
        const [dx, dy, dz] = this.direction_vector;
        const right_vector = [
            -dz * Math.sin(Math.PI / 2) + dy * Math.cos(Math.PI / 2),
            -dx * Math.cos(Math.PI / 2) + dz * Math.sin(Math.PI / 2),
            -dy * Math.sin(Math.PI / 2) + dx * Math.cos(Math.PI / 2)
        ];
        const norm = Math.sqrt(right_vector.reduce((acc, val) => acc + val * val, 0));
        const normalized_right_vector = norm > 0 ? right_vector.map(val => val / norm) : right_vector;
        this.position[X] -= normalized_right_vector[X] * distance;
        this.position[Y] -= normalized_right_vector[Y] * distance;
        this.position[Z] -= normalized_right_vector[Z] * distance;
    }

    rotate_left_right(angle) {
        this.rotation[YAW] += angle;
        this.direction_vector = this.calculate_direction_vector();
    }

    rotate_up_down(angle) {
        this.rotation[PITCH] += angle;
        this.direction_vector = this.calculate_direction_vector();
    }
}
