document.getElementById('predictionForm').addEventListener('submit', async (e) => {
    e.preventDefault();

    // Collect data from the form
    const formData = new FormData(e.target);
    const data = Object.fromEntries(formData.entries());

    try {
        // Send the input data to the back-end
        const response = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data),
        });

        // Handle the response
        const result = await response.json();
        document.getElementById('output').innerText = `
            Violent Crime: ${result.violent_crime}
            Non-Violent Crime: ${result.nonviolent_crime}
        `;
    } catch (error) {
        console.error('Error:', error);
        document.getElementById('output').innerText = 'Error making prediction.';
    }
});
