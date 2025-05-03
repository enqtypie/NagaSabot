import { Component, ViewChild, ElementRef } from '@angular/core';
import { CommonModule } from '@angular/common';
import { VideoResultComponent } from '../video-result/video-result.component';
import { VideoService } from '../services/video.service';
import { HeaderComponent } from '../header/header.component';
import { CameraRecorderComponent } from '../camera-recorder/camera-recorder.component';

@Component({
  selector: 'app-try-it-out',
  standalone: true,
  imports: [CommonModule, VideoResultComponent, HeaderComponent, CameraRecorderComponent],
  templateUrl: './try-it-out.component.html'
})
export class TryItOutComponent {
  @ViewChild('fileInput') fileInput!: ElementRef<HTMLInputElement>;

  videoBlob: Blob | null = null;
  showCameraModal = false;

  constructor(private videoService: VideoService) {}

  handleFileSelection(event: Event) {
    const input = event.target as HTMLInputElement;
    if (input.files && input.files[0]) {
      console.log('File selected:', input.files[0]);
      console.log('File size:', input.files[0].size);
      this.videoBlob = input.files[0];
      this.checkAndPredictVideo(input.files[0]);
    }
  }

  private checkAndPredictVideo(file: File) {
    const video = document.createElement('video');
    video.preload = 'metadata';
    video.src = URL.createObjectURL(file);
    video.onloadedmetadata = () => {
      URL.revokeObjectURL(video.src);
      if (video.duration < 1) {
        return;
      }
      this.predictVideo(file);
    };
  }

  private predictVideo(file: File) {
    console.log('Starting video prediction for file:', file.name, 'Size:', file.size);
    this.videoService.predictLipreading(file)
      .subscribe({
        next: ({ phrase, confidence }) => {
          console.log('Prediction successful:', { phrase, confidence });
        },
        error: (error) => {
          console.error('Prediction failed:', error);
        }
      });
  }

  handleVideoRecorded(videoBlob: Blob) {
    this.videoBlob = videoBlob;
    this.showCameraModal = false;
  }

  handleRestart() {
    console.log('Handling restart...');
    if (this.videoBlob) {
      console.log('Cleaning up video blob');
      this.videoBlob = null;
    }
    if (this.fileInput) {
      this.fileInput.nativeElement.value = '';
    }
  }
}