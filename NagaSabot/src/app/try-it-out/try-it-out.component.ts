import { Component, ViewChild, ElementRef, OnDestroy } from '@angular/core';
import { CommonModule } from '@angular/common';
import { VideoResultComponent } from '../video-result/video-result.component';
import { VideoService } from '../services/video.service';

@Component({
  selector: 'app-try-it-out',
  standalone: true,
  imports: [CommonModule, VideoResultComponent],
  templateUrl: './try-it-out.component.html',
  styleUrls: ['./try-it-out.component.css']
})
export class TryItOutComponent implements OnDestroy {
  @ViewChild('videoElement') videoElement!: ElementRef<HTMLVideoElement>;
  @ViewChild('fileInput') fileInput!: ElementRef<HTMLInputElement>;

  showCameraModal = false;
  isRecording = false;
  isModalClosing = false;
  videoBlob: Blob | null = null;
  private mediaStream: MediaStream | null = null;
  private mediaRecorder: MediaRecorder | null = null;
  private chunks: Blob[] = [];

  constructor(private videoService: VideoService) {}

  async openCameraModal() {
    this.showCameraModal = true;
    try {
      this.mediaStream = await navigator.mediaDevices.getUserMedia({ 
        video: { 
          width: { ideal: 1280 },
          height: { ideal: 720 }
        }
      });
      this.videoElement.nativeElement.srcObject = this.mediaStream;
    } catch (err) {
      console.error('Error accessing camera:', err);
      alert('Unable to access camera. Please make sure you have granted camera permissions.');
      this.closeCameraModal();
    }
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