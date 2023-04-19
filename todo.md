## TODO 

---

### Should be implemented

1. Return new id or new object from push 
1. Change push to insert & update
1. Batch operations support
1. Ignore mode - where all errors in data are logging and ignoring and all data without errors are processing
1. Progress display in GUI

### Desirable protocols and integrations

1. Zendesk 
1. Firebase (storage, settings)
1. BigCommerce 
1. Mongo 
1. MySql (or general sql db) 
1. Email 
1. Slack
1. Redis 
1. Jira
1. Yandex translate 
1. Faker 
1. Yandex storage
1. Zabbix
1. HTTP, WebDav, REST
1. FTP
1. YAML
1. Git
1. WebSocket client & server
1. Google tables
1. Logs parsing
1. System events

### Desirable features

1. Modular architecture
1. Web-interface
1. Speed control in gui mode
1. Add method to test connection to all endpoints
1. General endpoint and collection for the REST resource
1. Reports (pdf)
1. Add stream, generator, 'for await' notations support 


### Known problems

1. App freezes in GUI mode after executing

### Minor tasks
1. Rename 'read' to 'list'
1. Generator for unique id
1. Encoding problem solver
1. Make log functions interface as analog of console.log
1. Add filesystem endpoint method to create a collection for unique temp folder