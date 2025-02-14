# SvelteOpenCVWebsocket

This project demonstrates the integration of Svelte with TypeScript on the frontend and a C++-based backend application developed with Visual Studio. Communication between the frontend and backend is handled via WebSockets, with OpenCV used for image and video processing.

## Project Overview

- **Frontend**: Implemented with [Svelte](https://svelte.dev/) and TypeScript, providing a reactive user interface for displaying and interacting with image and video data.
- **Backend**: Developed in C++ using Visual Studio, handling image and video processing via [OpenCV](https://opencv.org/).
- **Communication**: Data transmission between frontend and backend occurs in real-time via WebSockets, enabling efficient and fast data processing.

## Prerequisites

Ensure the following components are installed on your system:

- **Node.js**: Required to run the frontend development server.
- **Visual Studio**: Needed for developing and building the C++ backend.
- **OpenCV**: Library for image and video processing in the backend.

## Installation and Execution

### Frontend

1. **Install dependencies**:

   Navigate to the `frontend` directory and run the following command:

   ```bash
   npm install
   ```

2. **Start the development server**:

   Start the development server with:

   ```bash
   npm run dev
   ```

   The server is accessible by default at `http://localhost:5000`.

### Backend

1. **Open the project**:

   Open the Visual Studio project in the `backend` directory.

2. **Ensure dependencies**:

   Make sure OpenCV is correctly integrated and all necessary libraries are available.

3. **Build and run**:

   Run the project in Visual Studio. The backend will listen for incoming WebSocket connections from the frontend.

## Usage

- **Real-time image processing**: The frontend sends image or video data via WebSockets to the backend, where it is processed using OpenCV and then sent back to the frontend.

- **Customizable processing**: Modify the image processing logic in the backend to fit your specific requirements by adjusting the relevant C++ functions.

## License

This project is licensed under the [MIT License](LICENSE).
