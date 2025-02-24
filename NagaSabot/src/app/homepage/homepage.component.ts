// home.component.ts
import { Component } from '@angular/core';
import { Router } from '@angular/router';
import { PermissionService } from '../../app/permission.service';
import { CommonModule } from '@angular/common';

interface FAQ {
  id: number;
  icon: string;
  question: string;
  answer: string;
}

@Component({
  selector: 'app-home',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './homepage.component.html'
})
export class HomeComponent {
  faqs: FAQ[] = [
    {
      id: 1,
      icon: '🤔',
      question: 'What is NagaSabot?',
      answer: 'NagaSabot is a cutting-edge platform that uses artificial intelligence to perform lip-reading in the Bikol-Naga dialect.'
    },
    {
      id: 2,
      icon: '💻',
      question: 'How does it work?',
      answer: 'Our system processes video input through advanced neural networks, combining CNN and LSTM technologies for accurate lip-reading.'
    },
    {
      id: 3,
      icon: '🔒',
      question: 'Is my data secure?',
      answer: 'Yes! We prioritize your privacy and security. All video processing happens locally on your device.'
    },
    {
      id: 4,
      icon: '📱',
      question: 'What devices are supported?',
      answer: 'NagaSabot works on most modern devices with a camera, including smartphones, tablets, and computers.'
    },
    {
      id: 5,
      icon: '🌐',
      question: 'Can I use it offline?',
      answer: 'Currently, NagaSabot requires an internet connection for initial loading, but core features work offline.'
    },
    {
      id: 6,
      icon: '🎯',
      question: 'How accurate is it?',
      answer: 'Our system achieves an average accuracy rate of 85% in controlled environments.'
    }
  ];
  
  constructor(
    private router: Router,
    private permissionService: PermissionService
  ) {}

  async requestCameraAccess() {
    const granted = await this.permissionService.requestCameraPermission();
    if (granted) {
      console.log('Camera permission granted');
    }
  }

  async requestStorageAccess() {
    const granted = await this.permissionService.requestStoragePermission();
    if (granted) {
      console.log('Storage permission granted');
    }
  }

  navigateToTryItOut() {
    this.router.navigate(['/try-it-out']);
  }

  scrollToAbout() {
    document.getElementById('about')?.scrollIntoView({ behavior: 'smooth' });
  }
}