document.getElementById('uploadForm').addEventListener('submit', async (event) => {
    event.preventDefault();
  
    const fileInput = document.getElementById('imageInput');
    if (!fileInput.files.length) {
      alert('Please select an image file first.');
      return;
    }
  
    // Prepare the form data to send to the FastAPI backend
    const formData = new FormData();
    formData.append('file', fileInput.files[0]); // name must match the parameter in the FastAPI endpoint
  
    try {
      // Adjust the URL if your FastAPI server is running elsewhere
      const response = await fetch('http://127.0.0.1:8000/predict', {
        method: 'POST',
        body: formData
      });
  
      if (!response.ok) {
        throw new Error('Server returned an error while predicting.');
      }
  
      const data = await response.json();
      const resultDiv = document.getElementById('result');
  
      if (data.error) {
        resultDiv.textContent = 'Error: ' + data.error;
      } else {
        // Display prediction results
        resultDiv.innerHTML = `
          <p><strong>Predicted Disaster Type:</strong> ${data.disaster_type}</p>
          <p><strong>Predicted Severity:</strong> ${data.severity}</p>
        `;
      }
    } catch (error) {
      console.error('Error:', error);
      alert('An error occurred while predicting the disaster type and severity.');
    }
  });
  