// WhatsApp send service for Sidenote.
//
// Connects directly to WhatsApp's multi-device protocol via whatsmeow --
// no browser, no screen-coordinate clicking, no keyboard focus stealing.
// One-time QR-code login on first run; the session then persists in
// whatsapp-session.db (in this same folder), so restarts don't need a
// re-scan.
//
// Runs as its own small HTTP server on localhost:8081. The Python
// FastAPI server (server.py / whatsapp_send.py) calls it with a plain
// POST request -- this Go process does not need to know anything about
// FastAPI, Groq, or the rest of the Python stack.
//
// STAGE 1 (this file, right now): confirm this connects and logs in.
// STAGE 2 (later): swap whatsapp_send.py to call POST /send instead of
// pywhatkit. Don't touch the Python side until stage 1 works cleanly.
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"syscall"

	_ "github.com/mattn/go-sqlite3"
	"github.com/mdp/qrterminal/v3"
	"go.mau.fi/whatsmeow"
	"go.mau.fi/whatsmeow/proto/waE2E"
	"go.mau.fi/whatsmeow/store/sqlstore"
	"go.mau.fi/whatsmeow/types"
	"go.mau.fi/whatsmeow/types/events"
	waLog "go.mau.fi/whatsmeow/util/log"
	"google.golang.org/protobuf/proto"
)

var client *whatsmeow.Client

// This service only sends -- it doesn't act on incoming messages. This
// handler exists because whatsmeow requires *an* event handler to be
// registered; incoming messages are just logged for visibility, not used.
func eventHandler(evt interface{}) {
	switch v := evt.(type) {
	case *events.Message:
		log.Printf("Received (ignored, this service only sends): %s", v.Message.GetConversation())
	}
}

// sendRequest is the JSON body Python posts to /send.
type sendRequest struct {
	Phone   string `json:"phone"`   // digits + country code, e.g. "31612345678" (leading + is stripped if present)
	Message string `json:"message"`
}

func sendHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "POST only", http.StatusMethodNotAllowed)
		return
	}
	var req sendRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "bad JSON body", http.StatusBadRequest)
		return
	}

	phone := strings.TrimPrefix(req.Phone, "+")
	jid := types.NewJID(phone, types.DefaultUserServer)

	_, err := client.SendMessage(context.Background(), jid, &waE2E.Message{
		Conversation: proto.String(req.Message),
	})
	if err != nil {
		http.Error(w, fmt.Sprintf("send failed: %v", err), http.StatusBadGateway)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "sent"})
}

func main() {
	dbLog := waLog.Stdout("Database", "INFO", true)
	container, err := sqlstore.New(context.Background(), "sqlite3", "file:whatsapp-session.db?_foreign_keys=on", dbLog)
	if err != nil {
		panic(err)
	}
	deviceStore, err := container.GetFirstDevice(context.Background())
	if err != nil {
		panic(err)
	}

	clientLog := waLog.Stdout("Client", "INFO", true)
	client = whatsmeow.NewClient(deviceStore, clientLog)
	client.AddEventHandler(eventHandler)

	if client.Store.ID == nil {
		// No session yet -- first run. Print a QR code to scan once.
		qrChan, _ := client.GetQRChannel(context.Background())
		if err = client.Connect(); err != nil {
			panic(err)
		}
		for evt := range qrChan {
			if evt.Event == "code" {
				fmt.Println("Scan this with WhatsApp -> Linked Devices:")
				qrterminal.GenerateHalfBlock(evt.Code, qrterminal.L, os.Stdout)
			} else {
				fmt.Println("Login event:", evt.Event)
			}
		}
	} else {
		// Session already persisted in whatsapp-session.db -- no QR needed.
		if err = client.Connect(); err != nil {
			panic(err)
		}
	}

	http.HandleFunc("/send", sendHandler)
	go func() {
		fmt.Println("WhatsApp send service listening on http://localhost:8081")
		if err := http.ListenAndServe("localhost:8081", nil); err != nil {
			log.Fatal(err)
		}
	}()

	// Keep the process (and the WhatsApp connection) alive until Ctrl+C.
	c := make(chan os.Signal, 1)
	signal.Notify(c, os.Interrupt, syscall.SIGTERM)
	<-c
	client.Disconnect()
}
