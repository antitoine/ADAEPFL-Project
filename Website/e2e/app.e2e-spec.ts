import { ADAEPFLProjectPage } from './app.po';

describe('adaepfl-project App', function() {
  let page: ADAEPFLProjectPage;

  beforeEach(() => {
    page = new ADAEPFLProjectPage();
  });

  it('should display message saying app works', () => {
    page.navigateTo();
    expect(page.getParagraphText()).toEqual('app works!');
  });
});
