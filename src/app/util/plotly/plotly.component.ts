import { Component, OnInit, Input } from '@angular/core';
import { JsonReaderService } from '../json-reader.service';
declare let Plotly:any;

@Component({
  selector: 'plotly',
  templateUrl: './plotly.component.html',
  styleUrls: ['./plotly.component.css']
})
export class PlotlyComponent implements OnInit {

  @Input('url') url: string;
  @Input('dim') dim: number = 1;
  @Input('labels') labels: string[] = [];

  constructor(private jsonReader: JsonReaderService) {}

  ngOnInit() {
    this.displayPlotlyGraph(this.url, this.dim, this.labels);
  }

  private displayPlotlyGraph(jsonUrl: string, dim=1, labels=[]) {
    this.jsonReader.readJsonData(jsonUrl)
      .subscribe(json => {
        if (dim == 1) {
          Plotly.newPlot('plotly', json);
        } else {
          Plotly.newPlot('plotly', json['All']['Time']);
        }

      });
  }
}
