// imageUpdate.js

let totalResponseTime = 0;
let numRequests = 0;
let averageResponseTime = 0;

export async function updateImage(data) {
    const timestamp = new Date().getTime();
    const imageUrl = `static/${data.image_file}?${timestamp}`;
    await urlUpdate(imageUrl);
    performance.mark('end');
    performance.measure('총 소요 시간', 'start', 'end');
    const responseTimes = performance.getEntriesByName('총 소요 시간');
    const responseTime = responseTimes[0].duration;
    totalResponseTime += responseTime;
    numRequests++;
    averageResponseTime = totalResponseTime / numRequests;
    console.log(`총 소요 시간: ${responseTime}ms`);
    console.log(`평균 응답 시간: ${averageResponseTime.toFixed(2)}ms`);
    performance.clearMarks();
    performance.clearMeasures();
    // updateImageStats(data.time, data.avg_time);
}

async function urlUpdate(imageUrl) {
    const imageElement = document.getElementById('image');
    // imageElement.src = imageUrl;
    return new Promise((resolve, reject) => {
        imageElement.onload = resolve;
        imageElement.onerror = reject;
        imageElement.src = imageUrl;
    });
}

export function updateImageStats(time, avgTime) {
    document.getElementById('time_val').textContent = time;
    document.getElementById('avg_time').textContent = avgTime;
}