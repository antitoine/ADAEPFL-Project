import {
  Component, Input, ViewEncapsulation, AfterViewInit, HostListener, OnChanges,
  SimpleChange
} from '@angular/core';
import { JsonReaderService } from '../json-reader.service';
import { UUID } from 'angular2-uuid';
declare let _:any;
//import * as _ from 'lodash';
declare let Plotly:any;

@Component({
  selector: 'plotly',
  templateUrl: './plotly.component.html',
  styleUrls: ['./plotly.component.css'],
  encapsulation: ViewEncapsulation.None
})
export class PlotlyComponent implements AfterViewInit, OnChanges {

  plotlyId: string = 'plotly-' + UUID.UUID();

  @Input('url') url: string;
  @Input('labels') labels: string[] = [];
  @Input('display') display: boolean = true;
  @Input('selected') selected: string[] = [];

  labelSelected: string[] = [];
  labelValues: string[][];

  schema: any;

  loading: boolean = false;
  loadingMsg: string;

  constructor(private jsonReader: JsonReaderService) {}

  // Need to wait that plotly container id are set in the DOM
  ngAfterViewInit() {
    this.setLoadingOn(0, 'Download data');
    this.jsonReader.readJson(this.url,
      () => this.loadingMsg = 'Decompress data',
      () => this.loadingMsg = 'Decode graph',
      () => this.loadingMsg = 'Generate graph',
    ).subscribe(json => {
        this.schema = json;
        if (this.labels.length == 0) {
          this.plot(json);
        } else {
          this.labelValues = new Array(this.labels.length);
          this.labelValues[0] = Object.keys(json).sort(PlotlyComponent.sortValues);
          this.preselectChart();
        }
        this.setLoadingOff();
      });
  }

  @HostListener('window:resize')
  onResize() {
    this.generateChart();
  }

  ngOnChanges(changes: {[propKey: string]: SimpleChange}) {
    if (changes.hasOwnProperty('display') && !changes['display'].previousValue && changes['display'].currentValue) {
      // TODO : Improve this fix, we need to wait that the view is checked before generate chart
      setTimeout(() => {
        this.generateChart();
      }, 500);
    }
  }

  onLabelSelectedChange(index: number, newValue: string, loading: boolean = true) {

    if (index < 0 || index >= this.labels.length) {
      console.error('Index out of range');
      return;
    }

    if (loading) {
      this.setLoadingOn();
    }

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
      this.labelValues[index+1] = Object.keys(subData).sort(PlotlyComponent.sortValues);

      if (loading) {
        this.setLoadingOff();
      }

    } else if (this.labels.length == this.labelSelected.length && _.every(this.labelSelected, (v, k) => this.hasLabelSelected(k))) {
      this.generateChart();
    }
  }

  hasLabelSelected(index: number) {
    return this.labelSelected.length > index && this.labelSelected[index];
  }

  private setLoadingOn(time: number = 0, msg: string = '') {
    this.loadingMsg = msg;
    this.loading = true;
    if (time > 0) {
      setTimeout(() => {
        this.setLoadingOff();
      }, time);
    }
  }

  private setLoadingOff() {
    this.loading = false;
    this.loadingMsg = '';
  }


  private static sortValues(a: any, b: any) {
    let upperA = a.toUpperCase();
    let upperB = b.toUpperCase();
    if (upperA.startsWith('ALL') && !upperB.startsWith('ALL')) {
      return -1;
    } else if (upperB.startsWith('ALL')) {
      return 1;
    }
    if (upperA < upperB) {
      return -1;
    }
    if (upperA > upperB) {
      return 1;
    }
    return 0;
  }

  private preselectChart() {
    for(let i = 0; i < this.labels.length; i++) {
      if (i < this.selected.length && this.labelValues[i].indexOf(this.selected[i]) >= 0) {
        this.onLabelSelectedChange(i, this.selected[i], false);
      } else {
        this.onLabelSelectedChange(i, this.labelValues[i][0], false);
      }
    }
  }

  private generateChart(loading:boolean = true) {
    if (loading) {
      this.setLoadingOn();
    }

    if (this.labels.length == 0) {

      this.plot(this.schema);

    } else if (this.labels.length == this.labelSelected.length && _.every(this.labelSelected, (v, k) => this.hasLabelSelected(k))) {


      // Check if all labels are selected in order to generate the Plotly graph
      let finalData = this.schema[this.labelSelected[0]];
      for (let i = 1; i < this.labels.length; i++) {
        finalData = finalData[this.labelSelected[i]];
      }
      this.plot(finalData);
    } else if (loading) {
      this.setLoadingOff();
      return;
    }

    if (loading) {
      this.setLoadingOn(500, 'Generate graph');
    }
  }

  private plot(data: any) {
    if (this.display && document.getElementById(this.plotlyId)) {
      try {
        Plotly.newPlot(this.plotlyId, data);
      } catch (e){
        console.error('Can display Plotly charts', e);
      }
    }
  }
}
