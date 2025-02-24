import { Component, Input, Output, EventEmitter } from '@angular/core';
import { CommonModule } from '@angular/common';

export interface VideoResult {
  videoUrl: string;
  phrase: string;
  accuracy: number;
  timestamp: number;
}

@Component({
  selector: 'app-video-result',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './video-result.component.html'
})
export class VideoResultComponent {
  @Input() result!: VideoResult;
  @Output() restart = new EventEmitter<void>();

  tryAgain() {
    this.restart.emit();
  }
} 