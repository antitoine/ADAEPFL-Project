import { Component, OnInit } from '@angular/core';
declare let Plotly:any;

@Component({
  selector: 'app-lausanne-1999-2016',
  templateUrl: './lausanne-1999-2016.component.html',
  styleUrls: ['./lausanne-1999-2016.component.css']
})
export class Lausanne19992016Component implements OnInit {

  data = [{
    values: [19, 26, 55],
    labels: ['Residential', 'Non-Residential', 'Utility'],
    type: 'pie'
  }];

  layout = {
    height: 380,
    width: 480
  };

  constructor() { }

  ngOnInit() {
    Plotly.newPlot('myDiv', this.data, this.layout);
  }

}
