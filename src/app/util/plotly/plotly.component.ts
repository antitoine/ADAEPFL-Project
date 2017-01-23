import { Component, OnInit, Input, ViewEncapsulation } from '@angular/core';
import { JsonReaderService } from '../json-reader.service';
declare let _:any;
//import * as _ from 'lodash';
declare let Plotly:any;

@Component({
  selector: 'plotly',
  templateUrl: './plotly.component.html',
  styleUrls: ['./plotly.component.css'],
  encapsulation: ViewEncapsulation.None
})
export class PlotlyComponent implements OnInit {

  @Input('url') url: string;
  @Input('labels') labels: string[] = [];

  labelSelected: string[] = [];
  labelValues: string[][];

  schema: any;

  loading: boolean = false;
  loadingTime: any;

  constructor(private jsonReader: JsonReaderService) {}

  ngOnInit() {
    this.loading = true;
    this.jsonReader.readJson(this.url)
      .subscribe(json => {
        this.schema = json;
        if (this.labels.length == 0) {
          Plotly.newPlot('plotly', json);
        } else {
          this.labelValues = new Array(this.labels.length);
          this.labelValues[0] = Object.keys(json);
        }
        this.loading = false;
      });
  }

  onLabelSelectedChange(index: number, label: string, newValue: string) {

    if (index < 0 || index >= this.labels.length) {
      console.error('Index out of range');
      return;
    }

    this.setLoading(300);

    // Remove upper old selected values
    for (let i = index + 1; i < this.labelSelected.length; i++) {
      this.labelSelected[i] = null;
      this.labelValues[i] = [];
    }

    // Set the new selected value
    this.labelSelected[index] = newValue;

    if (this.labels.length > index + 1) {

      // Populate upper selector
      let subData = this.schema[this.labelSelected[0]];
      for (let i = 1; i <= index; i++) {
        subData = subData[this.labelSelected[i]];
      }
      this.labelValues[index+1] = Object.keys(subData);

    } else if (this.labels.length == this.labelSelected.length && _.every(this.labelSelected, (v, k) => this.hasLabelSelected(k))) {

      // Check if all labels are selected in order to generate the Plotly graph
      let finalData = this.schema[this.labelSelected[0]];
      for (let i = 1; i < this.labels.length; i++) {
        finalData = finalData[this.labelSelected[i]];
      }
      Plotly.newPlot('plotly', finalData);
    }
  }

  hasLabelSelected(index: number) {
    return this.labelSelected.length > index && this.labelSelected[index];
  }

  private setLoading(time: number = 1000) {
    this.loading = true;
    this.loadingTime = setTimeout(() => {
      this.loading = false;
    }, time);
  }
}
