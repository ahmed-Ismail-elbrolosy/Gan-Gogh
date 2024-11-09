function submitForm() {
    const formData = new FormData();
    const fileInput = document.querySelector('input[type="file"]');

    if (fileInput.files.length > 0) {
        const file = fileInput.files[0];
        formData.append("file", file);

        fetch("http://127.0.0.1:5000/transform", {
            method: "POST",
            body: formData,
        })
            .then(response => {
                if (file.type.startsWith('image/')) {
                    return response.blob(); // Handle image response as Blob
                } else if (file.type.startsWith('video/')) {
                    return response.json(); // Handle video response as JSON
                }
            })
            .then(data => {
                if (file.type.startsWith('image/')) {
                    const fileURL = URL.createObjectURL(data);
                    console.log("Transformed File URL:", fileURL); // Log the file URL

                    const transformedImg = document.getElementById("transformed-img");
                    transformedImg.src = fileURL;
                    transformedImg.classList.add('show');
                    transformedImg.classList.remove('hidden');
                    document.getElementById("transformed-video").classList.remove('show');
                    document.getElementById("transformed-video").classList.add('hidden');

                    // Add click event to enlarge image
                    transformedImg.onclick = () => enlargeImage(fileURL);
                } else if (file.type.startsWith('video/')) {
                    const videoPath = data.output_video_path;
                    const fullOutputDir = data.full_output_dir;
                    console.log("Video Path:", videoPath);
                    console.log("Full Output Directory:", fullOutputDir);

                    const transformedVideo = document.getElementById("transformed-video");
                    transformedVideo.src = `http://127.0.0.1:5000/${videoPath}`;
                    transformedVideo.classList.add('show');
                    transformedVideo.classList.remove('hidden');
                    document.getElementById("transformed-img").classList.remove('show');
                    document.getElementById("transformed-img").classList.add('hidden');

                    // Add click event to enlarge video
                    transformedVideo.onclick = () => enlargeVideo(transformedVideo.src);
                }

                document.querySelector('.image-preview').classList.add('show'); // Show the preview container
            })
            .catch(error => console.error("Error:", error));
    } else {
        alert("Please select an image or video file.");
    }
}

function previewFile(event) {
    const file = event.target.files[0];
    const reader = new FileReader();
    reader.onload = function () {
        if (file.type.startsWith('image/')) {
            const output = document.getElementById('original-img');
            output.src = reader.result;
            output.classList.add('show');
            output.classList.remove('hidden');
            document.getElementById('original-video').classList.remove('show');
            document.getElementById('original-video').classList.add('hidden');

            // Add click event to enlarge image
            output.onclick = () => enlargeImage(reader.result);
        } else if (file.type.startsWith('video/')) {
            const output = document.getElementById('original-video');
            output.src = reader.result;
            output.classList.add('show');
            output.classList.remove('hidden');
            document.getElementById('original-img').classList.remove('show');
            document.getElementById('original-img').classList.add('hidden');

            // Add click event to enlarge video
            output.onclick = () => enlargeVideo(reader.result);
        }
        document.querySelector('.image-preview').classList.add('show'); // Show the preview container
    };
    reader.readAsDataURL(file);
}

function enlargeImage(src) {
    const enlargedContainer = document.createElement('div');
    enlargedContainer.classList.add('enlarged-container');
    enlargedContainer.onclick = () => document.body.removeChild(enlargedContainer);

    const enlargedImage = document.createElement('img');
    enlargedImage.src = src;
    enlargedImage.classList.add('enlarged-image');

    enlargedContainer.appendChild(enlargedImage);
    document.body.appendChild(enlargedContainer);

    setTimeout(() => {
        enlargedContainer.classList.add('show');
    }, 10);
}

function enlargeVideo(src) {
    const enlargedContainer = document.createElement('div');
    enlargedContainer.classList.add('enlarged-container');
    enlargedContainer.onclick = () => document.body.removeChild(enlargedContainer);

    const enlargedVideo = document.createElement('video');
    enlargedVideo.src = src;
    enlargedVideo.classList.add('enlarged-image');
    enlargedVideo.controls = true;
    enlargedVideo.autoplay = true;

    enlargedContainer.appendChild(enlargedVideo);
    document.body.appendChild(enlargedContainer);

    setTimeout(() => {
        enlargedContainer.classList.add('show');
    }, 10);
}