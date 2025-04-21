import { Component, ViewChild, ElementRef, OnDestroy, PLATFORM_ID, Inject, AfterViewInit } from '@angular/core';
import { CommonModule, isPlatformBrowser } from '@angular/common';
import { VideoResultComponent } from '../video-result/video-result.component';
import { VideoService } from '../services/video.service';

@Component({
  selector: 'app-try-it-out',
  standalone: true,
  imports: [CommonModule, VideoResultComponent],
  templateUrl: './try-it-out.component.html'
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
      const { FaceLandmarker, FilesetResolver, DrawingUtils } = await import('@mediapipe/tasks-vision');
      
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

      (this as any).FaceLandmarker = FaceLandmarker;
      (this as any).DrawingUtils = DrawingUtils;
    } catch (error) {
      console.error('Error initializing face landmarker:', error);
    }
  }

  async openCameraModal() {
    this.showCameraModal = true;

    try {
      if (!isPlatformBrowser(this.platformId)) {
        throw new Error('Camera access is only available in browser environments');
      }

      if (!navigator.mediaDevices?.getUserMedia) {
        throw new Error('getUserMedia is not supported in this browser');
      }

      const devices = await navigator.mediaDevices.enumerateDevices();
      const videoDevices = devices.filter(device => device.kind === 'videoinput');

      this.mediaStream = await navigator.mediaDevices.getUserMedia({ 
        video: { 
          width: { ideal: 640 },
          height: { ideal: 360 },
          facingMode: 'user',
          deviceId: videoDevices[0]?.deviceId
        }
      });

      await this.waitForViewInitialization();
      this.videoElement.nativeElement.srcObject = this.mediaStream;
      
      // Wait for video to be ready and playing
      await new Promise<void>((resolve) => {
        this.videoElement.nativeElement.onloadedmetadata = () => {
          this.videoElement.nativeElement.play();
          // Set canvas size to match video
          this.canvas.nativeElement.width = this.videoElement.nativeElement.videoWidth;
          this.canvas.nativeElement.height = this.videoElement.nativeElement.videoHeight;
          resolve();
        };
      });

      await this.startFaceTracking();
    } catch (err: any) {
      console.error('Camera error:', err);
      alert(this.getErrorMessage(err));
      this.closeCameraModal();
    }
  }

  private async waitForViewInitialization() {
    return new Promise<void>(resolve => {
      const checkView = () => {
        if (this.isViewInitialized && this.videoElement && this.canvas) {
          resolve();
        } else {
          setTimeout(checkView, 100);
        }
      };
      checkView();
    });
  }

  private async startFaceTracking() {
    if (!this.faceLandmarker || !this.videoElement?.nativeElement || !this.canvas?.nativeElement) return;

    const ctx = this.canvas.nativeElement.getContext('2d');
    if (!ctx) return;

    const drawingUtils = new (this as any).DrawingUtils(ctx);
    const FaceLandmarker = (this as any).FaceLandmarker;

    const detectFaces = async () => {
      try {
        if (this.videoElement.nativeElement.paused || this.videoElement.nativeElement.ended) {
          await this.videoElement.nativeElement.play();
        }

        // Only process if video time has changed
        if (this.videoElement.nativeElement.currentTime !== this.lastVideoTime) {
          this.lastVideoTime = this.videoElement.nativeElement.currentTime;
          const results = this.faceLandmarker.detectForVideo(
            this.videoElement.nativeElement,
            performance.now()
          );

          ctx.clearRect(0, 0, this.canvas.nativeElement.width, this.canvas.nativeElement.height);

          if (results.faceLandmarks) {
            for (const landmarks of results.faceLandmarks) {
              // Create a mirrored version of the landmarks
              const mirroredLandmarks = landmarks.map((point: { x: number; y: number; z: number }) => ({
                ...point,
                x: 1 - point.x // Mirror the x coordinate
              }));

              // Draw lip connectors with mirrored landmarks
              drawingUtils.drawConnectors(
                mirroredLandmarks,
                FaceLandmarker.FACE_LANDMARKS_LIPS,
                { color: "#FFFFFF", lineWidth: 2 }
              );

              // Draw lip points with mirrored landmarks
              const lipPoints = FaceLandmarker.FACE_LANDMARKS_LIPS.flat();
              for (const index of lipPoints) {
                const point = mirroredLandmarks[index];
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

  private getErrorMessage(error: any): string {
    const messages: { [key: string]: string } = {
      'NotAllowedError': 'Please make sure you have granted camera permissions.',
      'NotFoundError': 'No camera device found.',
      'NotReadableError': 'Camera is in use by another application.',
      'OverconstrainedError': 'Camera does not support the requested constraints.',
      'StreamApiNotSupportedError': 'Stream API is not supported in this browser.'
    };

    return `Unable to access camera. ${messages[error.name] || `Error: ${error.message || 'Unknown error occurred'}`}`;
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