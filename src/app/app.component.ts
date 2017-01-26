import { Component, OnInit, OnDestroy } from '@angular/core';
import {
  Router, NavigationEnd, NavigationStart, NavigationCancel, NavigationError,
  ActivatedRoute
} from '@angular/router';
import { Subscription } from 'rxjs/Rx';
import { SlimLoadingBarService } from 'ng2-slim-loading-bar';
declare let smoothScroll:any;


@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent implements OnInit, OnDestroy {

  menuToggled: boolean = false;

  routerEventsSubscription: Subscription;
  routeFragmentSubscription: Subscription;

  constructor(private router: Router, private slimLoader: SlimLoadingBarService, private route: ActivatedRoute) {}

  ngOnInit() {

    this.routerEventsSubscription = this.router.events
      .subscribe(event => {
        if (event instanceof NavigationStart) {
          this.slimLoader.start();
        } else if ( event instanceof NavigationEnd || event instanceof NavigationCancel || event instanceof NavigationError) {
          this.slimLoader.complete();
        }
        if (!(event instanceof NavigationEnd)) {
          return;
        }
      }, (error: any) => {
        this.slimLoader.complete();
      });

    this.routeFragmentSubscription = this.route.fragment
      .subscribe(fragment => {
        if (fragment) {
          let element = document.getElementById(fragment);
          if (element) {
            smoothScroll(element);
          }
        } else {
          smoothScroll(0);
        }
      });
  }

  ngOnDestroy() {
    this.routerEventsSubscription.unsubscribe();
    this.routeFragmentSubscription.unsubscribe();
  }
}
