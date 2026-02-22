document.addEventListener('DOMContentLoaded', function() {
    const symptomsInput = document.getElementById('symptoms');
    const sendBtn = document.getElementById('sendBtn');
    const resultOutput = document.getElementById('result');
    const loadingIndicator = document.getElementById('loading');
    const errorDiv = document.getElementById('error');
    
    // API endpoint URL - using relative path since frontend and backend are on same server
    const API_URL = '/diagnose';
    
    // Function to send symptoms to the API
    async function sendSymptoms() {
        const symptoms = symptomsInput.value.trim();
        
        // Show loading indicator
        loadingIndicator.classList.remove('hidden');
        errorDiv.classList.add('hidden');
        errorDiv.textContent = '';
        
        try {
            const response = await fetch(API_URL, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    symptoms: symptoms
                })
            });
            
            if (!response.ok) {
                throw new Error(`Ошибка HTTP: ${response.status}`);
            }
            
            const data = await response.json();
            
            // Format and display the JSON response
            resultOutput.textContent = JSON.stringify(data, null, 2);
            
        } catch (error) {
            console.error('Ошибка при отправке запроса:', error);
            errorDiv.textContent = `Ошибка: ${error.message}`;
            errorDiv.classList.remove('hidden');
        } finally {
            // Hide loading indicator
            loadingIndicator.classList.add('hidden');
        }
    }
    
    // Event listener for the Send button
    sendBtn.addEventListener('click', sendSymptoms);
    
    // Also allow sending with Enter key (Ctrl+Enter for new line in textarea)
    symptomsInput.addEventListener('keydown', function(event) {
        if (event.key === 'Enter' && !event.ctrlKey && !event.shiftKey) {
            event.preventDefault();
            sendSymptoms();
        }
    });
});