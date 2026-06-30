import streamlit.components.v1 as components

def render_bg_animation():
    html_code = """
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r134/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/vanta@latest/dist/vanta.net.min.js"></script>
    <script>
    // Execute after a short delay to ensure DOM is fully loaded
    setTimeout(() => {
        const parentDoc = window.parent.document;
        const body = parentDoc.body;
        
        if (body && !parentDoc.getElementById('vanta-bg')) {
            // Create a dedicated container for the background
            const bgContainer = parentDoc.createElement('div');
            bgContainer.id = 'vanta-bg';
            bgContainer.style.position = 'fixed';
            bgContainer.style.top = '0';
            bgContainer.style.left = '0';
            bgContainer.style.width = '100vw';
            bgContainer.style.height = '100vh';
            bgContainer.style.zIndex = '-999';
            
            // Insert it at the very beginning of the body
            body.insertBefore(bgContainer, body.firstChild);
            
            // Initialize Vanta 3D Net Animation
            VANTA.NET({
                el: bgContainer,
                mouseControls: true,
                touchControls: true,
                gyroControls: false,
                minHeight: 200.00,
                minWidth: 200.00,
                scale: 1.00,
                scaleMobile: 1.00,
                color: 0x38BDF8,          // Cyan nodes
                backgroundColor: 0x050816, // Dark background
                points: 12.00,
                maxDistance: 22.00,
                spacing: 18.00,
                showDots: true
            });
            
            // Make the Streamlit main containers transparent so the background shows through
            const stApp = parentDoc.querySelector('.stApp');
            if (stApp) stApp.style.background = 'transparent';
            
            const header = parentDoc.querySelector('header');
            if (header) header.style.background = 'transparent';
        }
    }, 1000); // 1 second delay to ensure Streamlit is ready
    </script>
    """
    components.html(html_code, height=0, width=0)
