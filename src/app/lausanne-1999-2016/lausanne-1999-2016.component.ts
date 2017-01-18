import { Component, OnInit } from '@angular/core';
import { JsonReaderService } from '../json-reader.service';
declare let Plotly:any;

@Component({
  selector: 'app-lausanne-1999-2016',
  templateUrl: './lausanne-1999-2016.component.html',
  styleUrls: ['./lausanne-1999-2016.component.css']
})
export class Lausanne19992016Component implements OnInit {

  constructor(private jsonReader: JsonReaderService) {}

  ngOnInit() {
    this.jsonReader.readJsonData('./assets/json/marathon-lausanne-1999-2016-ages-performances.json')
      .subscribe(json => {
        Plotly.newPlot('testPlotlyGraphFromNotebook', json['All']['Time']);
      });
  }
}
