import { Component, OnInit } from '@angular/core';

@Component({
  selector: 'app-lausanne-2016',
  templateUrl: './lausanne-2016.component.html',
  styleUrls: ['./lausanne-2016.component.css']
})
export class Lausanne2016Component implements OnInit {

  distributionByRunningType:string = '42';

  isDetailedStatisticalAnalysisCollapsed:boolean = true;

  constructor() {}

  ngOnInit() {
  }
}
