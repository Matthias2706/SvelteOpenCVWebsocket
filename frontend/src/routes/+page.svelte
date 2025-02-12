<script lang="ts">

    let frameCount:number = 0;
    let src:string = $state("data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAUAAAAFCAYAAACNbyblAAAAHElEQVQI12P4//8/w38GIAXDIBKE0DHxgljNBAAO9TXL0Y4OHwAAAABJRU5ErkJggg==");

</script>

<div class="grid h-dvh place-content-center">
    <img {src} class="size-64 bg-black" alt="Webcam">
    <p>Frame count: {frameCount}</p>
</div>