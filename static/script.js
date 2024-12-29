// Select elements
const form = document.getElementById("upload-form");
const resultImage = document.getElementById("result-image");

// Add event listener for form submission
form.addEventListener("submit", async (event) => {
    event.preventDefault(); // Prevent default form submission behavior

    const formData = new FormData(form); // Collect form data

    try {
        // Send files to the API
        const response = await fetch("/upload/", {
            method: "POST",
            body: formData,
        });

        if (response.ok) {
            // Display the returned result image
            const blob = await response.blob();
            resultImage.src = URL.createObjectURL(blob);
        } else {
            // Handle errors
            alert("Error uploading files. Please try again.");
        }
    } catch (error) {
        console.error("Error:", error);
        alert("An unexpected error occurred. Please try again.");
    }
});