import { Component, ViewChild, ElementRef, OnDestroy, PLATFORM_ID, Inject, AfterViewInit } from '@angular/core';
import { CommonModule, isPlatformBrowser } from '@angular/common';
import { VideoResultComponent } from '../video-result/video-result.component';
import { VideoService } from '../services/video.service';

@Component({
  selector: 'app-try-it-out',
  standalone: true,
  imports: [CommonModule, VideoResultComponent],
  templateUrl: './try-it-out.component.html',
  styleUrls: ['./try-it-out.component.css']
})
export class TryItOutComponent implements OnDestroy, AfterViewInit {
  @ViewChild('videoElement') videoElement!: ElementRef<HTMLVideoElement>;
  @ViewChild('fileInput') fileInput!: ElementRef<HTMLInputElement>;
  @ViewChild('canvas') canvas!: ElementRef<HTMLCanvasElement>;

  showCameraModal = false;
  isRecording = false;
  isModalClosing = false;
  videoBlob: Blob | null = null;
  private mediaStream: MediaStream | null = null;
  private mediaRecorder: MediaRecorder | null = null;
  private chunks: Blob[] = [];
  private faceLandmarker: any = null;
  private animationFrameId: number | null = null;
  private isViewInitialized = false;
  private lastVideoTime = -1;

  constructor(
    private videoService: VideoService,
    @Inject(PLATFORM_ID) private platformId: Object
  ) {}

  ngAfterViewInit() {
    this.isViewInitialized = true;
    this.initializeFaceLandmarker();
  }

  private async initializeFaceLandmarker() {
    try {
      // Import MediaPipe Tasks Vision
      const { FaceLandmarker, FilesetResolver, DrawingUtils } = await import('@mediapipe/tasks-vision');
      
      // Initialize the face landmarker
      const vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm"
      );
      
      this.faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task`,
          delegate: "CPU"
        },
        outputFaceBlendshapes: true,
        runningMode: "VIDEO",
        numFaces: 1
      });

      // Store FaceLandmarker class for static properties
      (this as any).FaceLandmarker = FaceLandmarker;
      // Store DrawingUtils for later use
      (this as any).DrawingUtils = DrawingUtils;

      console.log('Face landmarker initialized successfully');
    } catch (error) {
      console.error('Error initializing face landmarker:', error);
      throw new Error('Failed to initialize face landmarker');
    }
  }

  async openCameraModal() {
    this.showCameraModal = true;

    try {
      // First check if we're in a browser environment
      if (!isPlatformBrowser(this.platformId)) {
        throw new Error('Camera access is only available in browser environments');
      }

      // Check if getUserMedia is supported
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error('getUserMedia is not supported in this browser');
      }

      // List available video devices
      const devices = await navigator.mediaDevices.enumerateDevices();
      const videoDevices = devices.filter(device => device.kind === 'videoinput');
      console.log('Available video devices:', videoDevices);

      // Try to get camera access with specific constraints
      this.mediaStream = await navigator.mediaDevices.getUserMedia({ 
        video: { 
          width: { ideal: 640 },
          height: { ideal: 360 },
          facingMode: 'user',
          deviceId: videoDevices[0]?.deviceId
        }
      });

      // Wait for view to be initialized
      await new Promise<void>(resolve => {
        const checkView = () => {
          if (this.isViewInitialized && this.videoElement && this.canvas) {
            resolve();
          } else {
            setTimeout(checkView, 100);
          }
        };
        checkView();
      });

      // Set the video source
      this.videoElement.nativeElement.srcObject = this.mediaStream;
      
      // Remove mirroring from video element
      this.videoElement.nativeElement.style.transform = 'none';
      
      // Wait for video to be ready and playing
      await new Promise<void>((resolve) => {
        this.videoElement.nativeElement.onloadedmetadata = () => {
          this.videoElement.nativeElement.play();
          // Set canvas size to match video
          this.canvas.nativeElement.width = this.videoElement.nativeElement.videoWidth;
          this.canvas.nativeElement.height = this.videoElement.nativeElement.videoHeight;
          // Remove mirroring from canvas
          this.canvas.nativeElement.style.transform = 'none';
          resolve();
        };
      });

      // Start face tracking
      await this.startFaceTracking();
    } catch (err: any) {
      console.error('Detailed camera error:', err);
      let errorMessage = 'Unable to access camera. ';
      
      if (err.name === 'NotAllowedError') {
        errorMessage += 'Please make sure you have granted camera permissions.';
      } else if (err.name === 'NotFoundError') {
        errorMessage += 'No camera device found.';
      } else if (err.name === 'NotReadableError') {
        errorMessage += 'Camera is in use by another application.';
      } else if (err.name === 'OverconstrainedError') {
        errorMessage += 'Camera does not support the requested constraints.';
      } else if (err.name === 'StreamApiNotSupportedError') {
        errorMessage += 'Stream API is not supported in this browser.';
      } else {
        errorMessage += `Error: ${err.message || 'Unknown error occurred'}`;
      }
      
      alert(errorMessage);
      this.closeCameraModal();
    }
  }

  private async startFaceTracking() {
    if (!this.faceLandmarker || !this.videoElement?.nativeElement || !this.canvas?.nativeElement) return;

    const ctx = this.canvas.nativeElement.getContext('2d');
    if (!ctx) return;

    // Create drawing utilities
    const drawingUtils = new (this as any).DrawingUtils(ctx);
    const FaceLandmarker = (this as any).FaceLandmarker;

    const detectFaces = async () => {
      try {
        // Make sure video is playing
        if (this.videoElement.nativeElement.paused || this.videoElement.nativeElement.ended) {
          await this.videoElement.nativeElement.play();
        }

        let startTimeMs = performance.now();

        // Detect faces using detectForVideo
        if (this.videoElement.nativeElement.currentTime !== this.lastVideoTime) {
          this.lastVideoTime = this.videoElement.nativeElement.currentTime;
          const results = this.faceLandmarker.detectForVideo(
            this.videoElement.nativeElement,
            startTimeMs
          );

          // Clear canvas
          ctx.clearRect(0, 0, this.canvas.nativeElement.width, this.canvas.nativeElement.height);

          // Draw detections
          if (results.faceLandmarks) {
            for (const landmarks of results.faceLandmarks) {
              // Draw only the lips with a thicker, more visible line
              drawingUtils.drawConnectors(
                landmarks,
                FaceLandmarker.FACE_LANDMARKS_LIPS,
                { color: "#FFFFFF", lineWidth: 2 }
              );

              // Draw lip points for better visibility
              const lipPoints = FaceLandmarker.FACE_LANDMARKS_LIPS.flat();
              for (const index of lipPoints) {
                const point = landmarks[index];
                ctx.beginPath();
                ctx.arc(
                  point.x * this.canvas.nativeElement.width,
                  point.y * this.canvas.nativeElement.height,
                  2,
                  0,
                  2 * Math.PI
                );
                ctx.fillStyle = '#FF3030';
                ctx.fill();
              }
            }
          }
        }
      } catch (error) {
        console.error('Face tracking error:', error);
      }

      this.animationFrameId = requestAnimationFrame(detectFaces);
    };

    detectFaces();
  }

  closeCameraModal() {
    this.isModalClosing = true;
    setTimeout(() => {
      this.showCameraModal = false;
      this.isModalClosing = false;
      this.stopCamera();
    }, 300);
  }

  startRecording() {
    if (!this.mediaStream) return;

    this.chunks = [];
    this.isRecording = true;
    this.mediaRecorder = new MediaRecorder(this.mediaStream);

    this.mediaRecorder.ondataavailable = (event) => {
      if (event.data.size > 0) {
        this.chunks.push(event.data);
      }
    };

    this.mediaRecorder.onstop = () => {
      this.videoBlob = new Blob(this.chunks, { type: 'video/webm' });
      this.closeCameraModal();
    };

    this.mediaRecorder.start();
  }

  stopRecording() {
    if (this.isRecording && this.mediaRecorder) {
      this.mediaRecorder.stop();
      this.isRecording = false;
      this.closeCameraModal();
    }
  }

  stopCamera() {
    if (this.mediaStream) {
      this.mediaStream.getTracks().forEach(track => track.stop());
      this.mediaStream = null;
    }

    if (this.videoElement?.nativeElement) {
      this.videoElement.nativeElement.srcObject = null;
    }

    if (this.animationFrameId) {
      cancelAnimationFrame(this.animationFrameId);
      this.animationFrameId = null;
    }

    if (this.faceLandmarker) {
      this.faceLandmarker.dispose();
      this.faceLandmarker = null;
    }
  }

  handleFileSelection(event: Event) {
    const input = event.target as HTMLInputElement;
    if (input.files && input.files[0]) {
      this.videoBlob = input.files[0];
    }
  }

  handleRestart() {
    this.videoBlob = null;
    if (this.fileInput) {
      this.fileInput.nativeElement.value = '';
    }
  }

  ngOnDestroy() {
    this.stopCamera();
  }
}