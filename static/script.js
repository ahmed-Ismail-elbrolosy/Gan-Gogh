function submitForm() {
    const formData = new FormData();
    const fileInput = document.querySelector('input[type="file"]');

    if (fileInput.files.length > 0) {
        formData.append("image", fileInput.files[0]);

        fetch("http://127.0.0.1:5000/transform", {
            method: "POST",
            body: formData,
        })
            .then(response => response.blob())
            .then(data => {
                const imgURL = URL.createObjectURL(data);
                console.log("Transformed Image URL:", imgURL); // Log the image URL
                document.getElementById("transformed-img").src = imgURL;
                document.querySelector('.image-preview').classList.add('show'); // Show the image preview container
            })
            .catch(error => console.error("Error:", error));
    } else {
        alert("Please select an image file.");
    }
}

function previewImage(event) {
    const reader = new FileReader();
    reader.onload = function () {
        const output = document.getElementById('original-img');
        output.src = reader.result;
        document.querySelector('.image-preview').classList.add('show'); // Show the image preview container
    };
    reader.readAsDataURL(event.target.files[0]);
}