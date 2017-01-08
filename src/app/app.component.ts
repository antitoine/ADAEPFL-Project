import { Component, OnInit, OnDestroy } from '@angular/core';
import { Router, NavigationEnd } from '@angular/router';
import { Subscription } from 'rxjs/Rx';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent implements OnInit, OnDestroy {

  menuToggled: boolean = false;

  routerSubscription: Subscription;

  constructor(private router: Router) {}

  ngOnInit() {
    // Make scroll to top when changing route
    this.routerSubscription = this.router.events
      .subscribe(event => {
        if (!(event instanceof NavigationEnd)) {
          return;
        }
        window.scrollTo(0, 0);
      });
  }

  ngOnDestroy() {
    this.routerSubscription.unsubscribe();
  }
}
