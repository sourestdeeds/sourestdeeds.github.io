document.onkeydown = animateKeys;
        
        const colors = ['#ff0000', '#0ff000','#00ff00', '#0000ff', '#f0000f', '#fff000', '#000fff'];

        function animateKeys(event) {
            let keyCode = event.keyCode;
            let key = String.fromCharCode(keyCode);

            let keyElement = document.createElement('div');
            document.body.appendChild(keyElement);

            keyElement.style.position = 'absolute';

            let randX = Math.round(Math.random() * window.innerWidth);
            let randY = Math.round(Math.random() * window.innerHeight);

            keyElement.style.left = randX + 'px';
            keyElement.style.top = randY + 'px';
            keyElement.textContent = key;

            const color = colors[Math.floor(Math.random() * colors.length)];
            keyElement.style.color = color;

            keyElement.style.transition = 'all 0.5s linear 0s';
            keyElement.style.left = keyElement.offsetLeft - 30 + 'px';
            keyElement.style.top = keyElement.offsetTop - 30 + 'px';
            keyElement.style.fontSize = '100px';
            keyElement.style.opacity = 0;
        }