const form = document.getElementById("upload-form");
const resultDiv = document.getElementById("result");
const resultImage = document.getElementById("result-image");

form.addEventListener("submit", async (event) => {
    event.preventDefault();

    const formData = new FormData(form);

    try {
        const response = await fetch("/upload/", {
            method: "POST",
            body: formData,
        });

        if (response.ok) {
            const blob = await response.blob();
            const imageUrl = URL.createObjectURL(blob);

            // Display the result image
            resultImage.src = imageUrl;
            resultImage.style.display = "block";
        } else {
            alert("Error uploading files. Please try again.");
        }
    } catch (error) {
        console.error("Error:", error);
        alert("An unexpected error occurred. Please try again.");
    }
});