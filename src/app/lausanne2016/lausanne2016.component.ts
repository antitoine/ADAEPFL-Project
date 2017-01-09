import { Component, OnInit } from '@angular/core';

@Component({
  selector: 'app-lausanne2016',
  templateUrl: './lausanne2016.component.html',
  styleUrls: ['./lausanne2016.component.css']
})
export class Lausanne2016Component implements OnInit {
  public isDetailedStatisticalAnalysisCollapsed:boolean = true;
  public is10kmTukeyHSDTableCollapsed:boolean = true;
  public is21kmTukeyHSDTableCollapsed:boolean = true;
  public is42kmTukeyHSDTableCollapsed:boolean = true;

  constructor() { }

  ngOnInit() {
  }

}
