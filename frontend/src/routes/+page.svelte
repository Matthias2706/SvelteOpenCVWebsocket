<script lang="ts">

    import { onMount } from "svelte";
    import { slide,fade } from 'svelte/transition';

    let frameCount:number = $state(0);
    let src:string = $state("");

    let socket:WebSocket;

    onMount(() => {
        socket = new WebSocket("ws://localhost:80/camera/stream")
        socket.addEventListener("open", ()=> {
            console.log("Opened")
        })

        socket.addEventListener("message", (message: any) => {    
            frameCount++
            let imgData = "data:image/jpg;base64," + message.data;
            src = imgData;

            return false;
        });

        socket.addEventListener("close", (event: any) => {    
            console.log(event);
            src = "";
        });

        return ()=>{
            socket.close();
        };
    });

    function handleOnLoad()
    {
        width = img.width;
        height = img.height;
    }

    let width:number= $state(0);
    let height:number= $state(0);
    let img:HTMLImageElement;

</script>

<div class="grid h-dvh place-content-center drop-shadow-xl">
    <div class="rounded-xl overflow-hidden border-1 border-gray-400 max-w-128">

    {#if src.length > 0}
        <img bind:this={img} {src} class="bg-black" alt="Webcam" onload={handleOnLoad} transition:fade>
    {:else}
        <p>Loading stream...</p>
    {/if}
    <div class="p-1 rounded-b-lg bg-sky-600 text-white border-t-1">
        <p class="px-3">Frame count: {frameCount}</p>
        <p class="px-3">Resolution: {width} x {height}</p>
    </div>
    </div>
</div>