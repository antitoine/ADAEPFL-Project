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
  @Input('labels') labels: string[] = [];

  labelSelected: string[] = new Array(2);
  labelValues: string[][] = [];

  schema: any;

  loading: boolean = false;
  loadingTime: any;

  constructor(private jsonReader: JsonReaderService) {}

  ngOnInit() {
    this.loading = true;
    this.jsonReader.readJsonData(this.url)
      .subscribe(json => {
        this.schema = json;

        if (this.labels.length == 0) {
          Plotly.newPlot('plotly', json);

        } else if (0 < this.labels.length && this.labels.length < 3) {
          let firstLabelValues = Object.keys(json);
          this.labelValues.push(firstLabelValues);

          if (this.labels.length == 2) {
            this.labelValues.push(Object.keys(json[firstLabelValues[0]]));
          }

        } else {
          console.error('Plotly module helper doesn\'t handle more than 3 dimensions (2 labels).')
        }
        this.setLoading();
      });
  }

  onLabelSelectedChange(index: number, label: string, newValue: string) {
    this.loading = true;
    this.labelSelected[index] = newValue;
    if (this.labels.length > 0 && this.labelSelected[0]) {

      if (this.labels.length == 1) {
        Plotly.newPlot('plotly',
          this.schema[this.labelSelected[this.labels[0]]]
        );

      } else if (this.labels.length == 2 && this.labelSelected[1]) {

        Plotly.newPlot('plotly',
          this.schema[this.labelSelected[0]][this.labelSelected[1]]
        );
      }
      this.setLoading(200);
    }
  }

  private setLoading(time: number = 1000) {
    this.loading = true;
    this.loadingTime = setTimeout(() => {
      this.loading = false;
    }, time);
  }
}
