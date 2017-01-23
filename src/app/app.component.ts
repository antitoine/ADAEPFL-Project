import { Component, OnInit, OnDestroy } from '@angular/core';
import { Router, NavigationEnd, NavigationStart, NavigationCancel, NavigationError } from '@angular/router';
import { Subscription } from 'rxjs/Rx';
import { SlimLoadingBarService } from 'ng2-slim-loading-bar';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent implements OnInit, OnDestroy {

  menuToggled: boolean = false;

  routerSubscription: Subscription;

  constructor(private router: Router, private slimLoader: SlimLoadingBarService) {}

  ngOnInit() {
    // Make scroll to top when changing route
    this.routerSubscription = this.router.events
      .subscribe(event => {
        if (event instanceof NavigationStart) {
          this.slimLoader.start();
        } else if ( event instanceof NavigationEnd || event instanceof NavigationCancel || event instanceof NavigationError) {
          this.slimLoader.complete();
        }
        if (!(event instanceof NavigationEnd)) {
          return;
        }
        window.scrollTo(0, 0);
      }, (error: any) => {
        this.slimLoader.complete();
      });
  }

  ngOnDestroy() {
    this.routerSubscription.unsubscribe();
  }
}
